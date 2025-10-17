# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class SharedTensorHandle:
    shm_name: str
    shape: Tuple[int, ...]
    dtype: str

    def to_shared_tensor(self) -> SharedTensor:
        """
        Create a SharedTensor from this handle.

        Returns:
            SharedTensor instance attached to the shared memory referenced by this handle
        """
        return SharedTensor(handle=self)

    def drop(self) -> None:
        """
        Unlink the shared memory segment.

        This marks the shared memory for deletion. The actual memory will be freed
        once all processes have closed their handles to it.

        Note: This only unlinks, it does not close any handles. Processes that have
        opened this shared memory should call close() on their SharedTensor instances.
        """
        try:
            # Attach to the shared memory just to unlink it
            shm = shared_memory.SharedMemory(name=self.shm_name)
            shm.close()
            shm.unlink()
        except Exception:
            pass


class SharedTensor:
    """
    Wrapper class for tensors backed by shared memory.

    This class provides a way to share tensors between processes using POSIX shared memory.
    It's designed for efficient inter-process tensor communication without copying data.

    Ownership and Lifecycle Model:
    ------------------------------
    1. **Creator process**:
       - Creates SharedTensor with tensor data or empty
       - Gets a handle via get_handle() to pass to other processes
       - **MUST** call close() after getting handle to release its reference
       - **SHOULD** call drop()/unlink() when all processes are done

    2. **Receiver processes**:
       - Receive SharedTensorHandle (via RPC, pickle, etc.)
       - Create SharedTensor from handle: SharedTensor(handle=handle)
       - Use the tensor: handle.to_shared_tensor().tensor
       - **MUST** call close() when done using the tensor

    3. **Cleanup**:
       - close(): Closes this process's file descriptor/handle
       - drop()/unlink(): Marks shared memory for deletion (call once, from any process)
       - Actual memory is freed when all processes have closed AND unlink is called

    Memory Leak Prevention:
    ----------------------
    - **DO NOT** rely on __del__ for cleanup! Python GC is unpredictable.
    - **ALWAYS** explicitly call close() when done with a SharedTensor
    - **ALWAYS** call drop() on handles when sharing is complete
    - Use context manager (with statement) for automatic cleanup
    - After close(), accessing .tensor will raise RuntimeError
    - After close(), getting handle will raise RuntimeError

    Closed State Behavior:
    ---------------------
    - Once close() is called, the SharedTensor enters a closed state
    - Accessing .tensor after close() raises RuntimeError
    - Calling get_handle() after close() raises RuntimeError
    - You can check the state with the .is_closed property
    - close() and drop() are idempotent (safe to call multiple times)

    Important Warning:
    ------------------
    If you hold a reference to the tensor BEFORE calling close(), that
    reference becomes INVALID after close():
        t = shared.tensor  # Get reference
        shared.close()     # Close SharedTensor - unmaps memory
        t.sum()            # SEGFAULT! The memory is now invalid

    After close(), the shared memory mapping is unmapped, so ALL references
    to the tensor (including cached ones) point to invalid memory. Accessing
    them will cause segmentation faults or undefined behavior.

    Always ensure you're done with the tensor before calling close().

    Example Usage:
    -------------
    # Creator process
    tensor = torch.randn(100, 100)
    shared = SharedTensor(tensor=tensor)
    handle = shared.get_handle()
    shared.close()  # Close creator's reference
    # ... send handle to other process via RPC ...
    handle.drop()  # Unlink after all receivers have it

    # Receiver process
    # ... receive handle via RPC ...
    shared = SharedTensor(handle=handle)
    result = shared.tensor.sum()  # Use the tensor
    shared.close()  # Close receiver's reference

    # Or use context manager (recommended)
    with SharedTensor(handle=handle) as shared:
        result = shared.tensor.sum()
    # Automatically closed
    """

    def __init__(
        self,
        *,
        tensor: Optional[torch.Tensor] = None,
        handle: Optional[SharedTensorHandle] = None,
    ):
        if tensor is not None:
            self._create_from_tensor(tensor)
        elif handle is not None:
            self._create_from_handle(handle)
        else:
            raise ValueError("Must provide either tensor or handle")

    @classmethod
    def empty(
        cls,
        shape: Union[Tuple[int, ...], torch.Size],
        dtype: torch.dtype = torch.float32,
    ):
        """
        Create an empty tensor directly in shared memory (no copy/allocation overhead)

        Args:
            shape: Shape of the tensor
            dtype: PyTorch dtype (supports bfloat16, float32, etc.)

        Returns:
            SharedTensor instance with uninitialized data
        """
        instance = cls.__new__(cls)
        instance._create_empty(shape, dtype)
        return instance

    @classmethod
    def zeros(
        cls,
        shape: Union[Tuple[int, ...], torch.Size],
        dtype: torch.dtype = torch.float32,
    ):
        """
        Create a zero-initialized tensor in shared memory

        Args:
            shape: Shape of the tensor
            dtype: PyTorch dtype

        Returns:
            SharedTensor instance with zeros
        """
        shared_tensor = cls.empty(shape, dtype)
        shared_tensor.tensor.zero_()
        return shared_tensor

    @classmethod
    def ones(
        cls,
        shape: Union[Tuple[int, ...], torch.Size],
        dtype: torch.dtype = torch.float32,
    ):
        """
        Create a ones-initialized tensor in shared memory

        Args:
            shape: Shape of the tensor
            dtype: PyTorch dtype

        Returns:
            SharedTensor instance with ones
        """
        shared_tensor = cls.empty(shape, dtype)
        shared_tensor.tensor.fill_(1)
        return shared_tensor

    def _create_empty(self, shape, dtype):
        """Initialize with empty tensor in shared memory"""
        # Initialize lifecycle state
        self._closed = False
        self._tensor_cache = None

        # Store metadata
        self._shape = tuple(shape) if not isinstance(shape, tuple) else shape
        self._dtype = dtype
        self._dtype_str = str(dtype)

        # Calculate size
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_elements = int(np.prod(self._shape))
        byte_size = total_elements * element_size

        # Create shared memory (uninitialized - fast!)
        shm_name = f"shared_tensor_{uuid.uuid4().hex}"
        self._shm = shared_memory.SharedMemory(
            create=True, size=byte_size, name=shm_name
        )
        self._shm_name = shm_name

    def _create_from_tensor(self, tensor):
        """Initialize from an existing tensor"""
        # Initialize lifecycle state
        self._closed = False
        self._tensor_cache = None

        tensor = tensor.contiguous()

        # Store metadata
        self._shape = tuple(tensor.shape)
        self._dtype = tensor.dtype
        self._dtype_str = str(tensor.dtype)

        # Create shared memory
        byte_size = tensor.numel() * tensor.element_size()
        shm_name = f"shared_tensor_{uuid.uuid4().hex}"

        self._shm = shared_memory.SharedMemory(
            create=True, size=byte_size, name=shm_name
        )
        self._shm_name = shm_name

        # Copy data as raw bytes
        raw_bytes = tensor.view(torch.uint8).view(-1).cpu().contiguous().numpy()
        self._shm.buf[:byte_size] = raw_bytes

    def _create_from_handle(self, handle: SharedTensorHandle):
        """Initialize from a handle"""
        # Initialize lifecycle state
        self._closed = False
        self._tensor_cache = None

        self._shm_name = handle.shm_name
        self._shape = handle.shape
        self._dtype_str = handle.dtype
        self._dtype = self._parse_dtype(self._dtype_str)

        # Attach to existing shared memory\
        self._shm = shared_memory.SharedMemory(name=self._shm_name)

    def _create_tensor_view(self):
        """Create tensor view of shared memory."""
        element_size = torch.tensor([], dtype=self._dtype).element_size()
        total_elements = int(np.prod(self._shape))
        byte_size = total_elements * element_size

        # Create numpy array that shares the buffer
        np_array = np.ndarray(shape=(byte_size,), dtype=np.uint8, buffer=self._shm.buf)
        # Create torch tensor from numpy (shares memory)
        uint8_tensor = torch.from_numpy(np_array)
        tensor = uint8_tensor.view(self._dtype).reshape(self._shape)

        # Keep both the np array and the SharedTensor object alive
        tensor._forge_shared_tensor = self
        tensor._forge_np_array = np_array

        return tensor

    def _parse_dtype(self, dtype_str):
        """Parse dtype string"""
        dtype_str = dtype_str.replace("torch.", "")
        return getattr(torch, dtype_str)

    def get_handle(self):
        """
        Get a picklable handle to share this SharedTensor with other processes.

        Returns:
            SharedTensorHandle: A lightweight handle that can be pickled and sent to other processes

        Raises:
            RuntimeError: If called after close() has been called
        """
        if self._closed:
            raise RuntimeError(
                "Cannot get handle after close(). Get the handle before closing."
            )
        return SharedTensorHandle(
            shm_name=self._shm_name,
            shape=self._shape,
            dtype=self._dtype_str,
        )

    @property
    def tensor(self):
        """
        Get the underlying tensor.

        Returns:
            torch.Tensor: View into the shared memory

        Raises:
            RuntimeError: If accessed after close() has been called
        """
        if self._closed:
            raise RuntimeError(
                "Cannot access tensor after close(). The SharedTensor has been closed."
            )
        if self._tensor_cache is None:
            self._tensor_cache = self._create_tensor_view()
        return self._tensor_cache

    def copy_from(self, source_tensor):
        """
        Copy data from another tensor into this shared tensor
        Useful when you create empty tensor first, then fill it

        Args:
            source_tensor: Source tensor to copy from
        """
        if source_tensor.shape != self._shape:
            raise ValueError(f"Shape mismatch: {source_tensor.shape} vs {self._shape}")
        # Copy data
        self.tensor.copy_(source_tensor)

    def clone(self):
        """Create a new SharedTensor with copied data"""
        new_shared = SharedTensor.empty(self._shape, self._dtype)
        new_shared.tensor.copy_(self.tensor)
        return new_shared

    def close(self):
        """
        Close this process's handle to the shared memory.

        This should be called when this process is done using the shared memory.
        The shared memory will persist until all processes have closed their handles
        and someone calls unlink().

        After calling close(), this SharedTensor object should not be used anymore.
        Accessing the tensor property after close() will raise a RuntimeError.

        This method is idempotent - calling it multiple times is safe.

        Note: If you hold a reference to the tensor before calling close(),
        that reference will remain valid, but new accesses via shared.tensor
        will raise an error.
        """
        if self._closed:
            return  # Already closed, nothing to do

        self._closed = True
        self._tensor_cache = None  # Release tensor and numpy array references

        try:
            self._shm.close()
        except Exception:
            pass

    def drop(self):
        """
        Close and unlink the shared memory.

        This method first closes this process's handle (if not already closed),
        then marks the shared memory for deletion. The actual memory will be freed
        once all processes have closed their handles.

        This method is idempotent - calling it multiple times is safe.

        Note:
            This should be called when the shared tensor is no longer needed.
            Failing to call this method may result in shared memory leaks.
        """
        # Close first to set _closed flag and release cache
        self.close()

        # Then unlink
        try:
            self._shm.unlink()
        except Exception:
            pass

    @property
    def is_closed(self) -> bool:
        """
        Check if this SharedTensor has been closed.

        Returns:
            bool: True if close() has been called, False otherwise
        """
        return self._closed

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes the shared memory handle."""
        self.close()
        return False

    def __del__(self):
        """
        Best-effort cleanup on garbage collection.

        WARNING: Do NOT rely on __del__ for cleanup! Python's garbage collector
        may not call __del__ promptly or at all, which can cause memory leaks.
        Always explicitly call close() when done with the SharedTensor.

        This __del__ is only a safety net for cases where explicit cleanup is missed.
        """
        # Only close if the object was fully initialized
        if hasattr(self, "_closed"):
            self.close()

    def __repr__(self):
        return f"SharedTensor(shape={self._shape}, dtype={self._dtype}, shm_name={self._shm_name})"
