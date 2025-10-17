# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import time

import pytest
import torch

# Assuming SharedTensor is in shared_tensor.py
from forge.util._shared_tensor import SharedTensor
from multiprocess import Process, Queue


class TestSharedTensorCreation:
    """Test tensor creation methods"""

    def test_empty_creation(self):
        """Test creating empty tensor"""
        shape = (100, 200)
        dtype = torch.float32

        shared = SharedTensor.empty(shape, dtype)

        assert shared.tensor.shape == torch.Size(shape)
        assert shared.tensor.dtype == dtype
        assert shared.tensor.shape == torch.Size(shape)
        assert shared.tensor.dtype == dtype

        shared.drop()

    def test_empty_with_bfloat16(self):
        """Test creating empty bfloat16 tensor"""
        shape = (50, 50)
        shared = SharedTensor.empty(shape, torch.bfloat16)

        assert shared.tensor.dtype == torch.bfloat16
        assert shared.tensor.dtype == torch.bfloat16

        shared.drop()

    def test_zeros_creation(self):
        """Test creating zero-initialized tensor"""
        shape = (10, 20)
        shared = SharedTensor.zeros(shape, torch.float32)

        tensor = shared.tensor
        assert torch.all(tensor == 0)
        assert tensor.sum().item() == 0.0

        shared.drop()

    def test_ones_creation(self):
        """Test creating ones-initialized tensor"""
        shape = (10, 20)
        shared = SharedTensor.ones(shape, torch.float32)

        tensor = shared.tensor
        assert torch.all(tensor == 1)
        assert tensor.sum().item() == 200.0

        shared.drop()

    def test_from_tensor_creation(self):
        """Test creating from existing tensor"""
        original = torch.randn(50, 50)
        shared = SharedTensor(tensor=original)

        assert shared.tensor.shape == original.shape
        assert shared.tensor.dtype == original.dtype
        assert torch.allclose(shared.tensor, original)

        shared.drop()

    def test_from_handle_creation(self):
        """Test creating from handle"""
        # Create original
        original = SharedTensor.empty((10, 10), torch.float32)
        original.tensor.fill_(5.0)

        # Get handle
        handle = original.get_handle()

        # Create from handle
        reconstructed = SharedTensor(handle=handle)

        assert torch.all(reconstructed.tensor == 5.0)
        assert reconstructed.tensor.shape == original.tensor.shape
        assert reconstructed.tensor.dtype == original.tensor.dtype

        original.drop()

    def test_creation_requires_argument(self):
        """Test that creation without arguments raises error"""
        with pytest.raises(ValueError, match="Must provide either tensor or handle"):
            SharedTensor()

    @pytest.mark.parametrize(
        "shape",
        [
            (10,),
            (10, 20),
            (5, 10, 15),
            (2, 3, 4, 5),
        ],
    )
    def test_various_shapes(self, shape):
        """Test creation with various shapes"""
        shared = SharedTensor.empty(shape, torch.float32)
        assert shared.tensor.shape == torch.Size(shape)
        assert shared.tensor.shape == torch.Size(shape)
        shared.drop()


class TestSharedTensorDtypes:
    """Test all supported dtypes"""

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        ],
    )
    def test_all_dtypes(self, dtype):
        """Test that all dtypes work correctly"""
        shape = (10, 10)
        shared = SharedTensor.empty(shape, dtype)

        assert shared.tensor.dtype == dtype
        assert shared.tensor.dtype == dtype

        # Test that we can write to it
        if dtype == torch.bool:
            shared.tensor.fill_(True)
        elif dtype in [torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8]:
            shared.tensor.fill_(42)
        else:
            shared.tensor.fill_(3.14)

        shared.drop()

    def test_dtype_conversion_in_handle(self):
        """Test dtype is preserved through handle"""
        for dtype in [torch.float32, torch.bfloat16, torch.int64]:
            shared1 = SharedTensor.empty((5, 5), dtype)
            handle = shared1.get_handle()

            shared2 = SharedTensor(handle=handle)
            assert shared2.tensor.dtype == dtype

            shared1.drop()


class TestSharedTensorOperations:
    """Test tensor operations"""

    def test_copy_from(self):
        """Test copying data from another tensor"""
        source = torch.randn(20, 30)
        shared = SharedTensor.empty((20, 30), torch.float32)

        shared.copy_from(source)

        assert torch.allclose(shared.tensor, source)
        shared.drop()

    def test_copy_from_shape_mismatch(self):
        """Test copy_from raises error on shape mismatch"""
        source = torch.randn(10, 10)
        shared = SharedTensor.empty((20, 20), torch.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            shared.copy_from(source)

        shared.drop()

    def test_clone(self):
        """Test cloning creates independent copy"""
        original = SharedTensor.empty((10, 10), torch.float32)
        original.tensor.fill_(5.0)

        cloned = original.clone()

        # Verify data is same
        assert torch.all(cloned.tensor == 5.0)

        # Verify they're independent
        original.tensor.fill_(10.0)
        assert torch.all(cloned.tensor == 5.0)
        assert torch.all(original.tensor == 10.0)

        original.drop()
        cloned.drop()

    def test_tensor_modifications(self):
        """Test that modifications to tensor are reflected"""
        shared = SharedTensor.zeros((10, 10), torch.float32)
        tensor = shared.tensor

        tensor[0, 0] = 99.0
        tensor[5:, :] = 42.0

        # Get tensor again and verify changes persist
        tensor2 = shared.tensor
        assert tensor2[0, 0].item() == 99.0
        assert torch.all(tensor2[5:, :] == 42.0)

        shared.drop()

    def test_inplace_operations(self):
        """Test in-place operations work"""
        shared = SharedTensor.empty((100, 100), torch.float32)
        tensor = shared.tensor

        tensor.normal_(0, 1)
        mean = tensor.mean().item()

        tensor.add_(5.0)
        new_mean = tensor.mean().item()

        assert abs(new_mean - (mean + 5.0)) < 0.1

        shared.drop()


class TestSharedTensorSerialization:
    """Test pickling and handle serialization"""

    def test_handle_is_picklable(self):
        """Test that handle can be pickled"""
        shared = SharedTensor.empty((10, 10), torch.float32)
        handle = shared.get_handle()

        # Pickle and unpickle
        pickled = pickle.dumps(handle)
        unpickled_handle = pickle.loads(pickled)

        assert unpickled_handle == handle

        shared.drop()

    def test_handle_small_size(self):
        """Test that handle is small (efficient for RPC)"""
        shared = SharedTensor.empty((10000, 10000), torch.float32)
        handle = shared.get_handle()

        pickled = pickle.dumps(handle)

        # Handle should be < 1KB even for huge tensors
        assert len(pickled) < 1024

        shared.drop()

    def test_data_integrity_after_pickle(self):
        """Test data is preserved through handle pickling"""
        # Create and fill tensor
        shared1 = SharedTensor.empty((50, 50), torch.bfloat16)
        shared1.tensor.normal_(0, 1)
        original_data = shared1.tensor.clone()

        # Pickle handle
        handle = shared1.get_handle()
        pickled = pickle.dumps(handle)
        unpickled_handle = pickle.loads(pickled)

        # Reconstruct
        shared2 = SharedTensor(handle=unpickled_handle)

        # Verify data is same
        assert torch.allclose(shared2.tensor.float(), original_data.float(), rtol=1e-3)

        shared1.drop()


class TestSharedTensorMemory:
    """Test memory management and cleanup"""

    def test_drop(self):
        """Test drop removes shared memory"""
        shared = SharedTensor.empty((10, 10), torch.float32)
        shm_name = shared._shm_name

        # Verify shared memory exists
        tensor = shared.tensor
        tensor.fill_(5.0)

        # Drop shared memory
        shared.drop()

        # Trying to attach should fail
        from multiprocessing import shared_memory

        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=shm_name)

    def test_multiple_views_same_memory(self):
        """Test multiple tensor views point to same memory"""
        shared = SharedTensor.empty((10, 10), torch.float32)

        tensor1 = shared.tensor
        tensor1.fill_(5.0)

        tensor2 = shared.tensor
        assert torch.all(tensor2 == 5.0)

        # Modify through tensor2
        tensor2.fill_(10.0)

        # Verify tensor1 sees the change
        assert torch.all(tensor1 == 10.0)

        shared.drop()

    def test_handle_reconstruction_shares_memory(self):
        """Test that handle reconstruction shares same memory"""
        shared1 = SharedTensor.empty((20, 20), torch.float32)
        shared1.tensor.fill_(7.0)

        handle = shared1.get_handle()
        shared2 = SharedTensor(handle=handle)

        # Modify through shared2
        shared2.tensor.fill_(14.0)

        # Verify shared1 sees the change
        assert torch.all(shared1.tensor == 14.0)

        shared1.drop()


class TestSharedTensorEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_shape(self):
        """Test scalar tensor (empty shape)"""
        shared = SharedTensor.ones((), torch.float32)
        assert shared.tensor.shape == ()
        assert shared.tensor.numel() == 1
        assert torch.allclose(
            shared.tensor,
            torch.ones(
                (),
            ),
        )
        shared.drop()

    def test_single_element_tensor(self):
        """Test 1-element tensor"""
        shared = SharedTensor.empty((1,), torch.float32)
        shared.tensor.fill_(42.0)
        assert shared.tensor.item() == 42.0
        shared.drop()

    def test_large_tensor(self):
        """Test large tensor (1GB)"""
        # 1GB tensor: 250M float32 elements
        shape = (250_000_000,)
        shared = SharedTensor.empty(shape, torch.float32)

        assert shared.tensor.shape == shape
        assert shared.tensor.numel() == 250_000_000

        shared.drop()

    def test_non_contiguous_tensor_conversion(self):
        """Test that non-contiguous tensors are handled"""
        # Create non-contiguous tensor
        original = torch.randn(10, 10).t()  # Transpose makes it non-contiguous
        assert not original.is_contiguous()

        # Should work (internally makes contiguous)
        shared = SharedTensor(tensor=original)

        # Result should match
        assert torch.allclose(shared.tensor, original)

        shared.drop()

    def test_repr(self):
        """Test string representation"""
        shared = SharedTensor.empty((10, 20), torch.float32)
        repr_str = repr(shared)

        assert "SharedTensor" in repr_str
        assert "10, 20" in repr_str
        assert "float32" in repr_str
        assert shared._shm_name in repr_str

        shared.drop()


class TestSharedTensorMultiprocess:
    """Test multiprocess scenarios"""

    def test_multiprocess_read(self):
        """Test reading shared tensor from another process"""

        def reader_process(handle_dict, result_queue):
            shared = SharedTensor(handle=handle_dict)
            tensor = shared.tensor
            result_queue.put(tensor.sum().item())

        # Create shared tensor in main process
        shared = SharedTensor.empty((100, 100), torch.float32)
        shared.tensor.fill_(5.0)

        # Read from child process
        result_queue = Queue()
        handle = shared.get_handle()

        p = Process(target=reader_process, args=(handle, result_queue))
        p.start()
        p.join()

        result = result_queue.get()
        expected = 5.0 * 100 * 100

        assert abs(result - expected) < 1e-5

        shared.drop()

    def test_multiprocess_write(self):
        """Test writing to shared tensor from another process"""

        def writer_process(handle_dict, value):
            shared = SharedTensor(handle=handle_dict)
            shared.tensor.fill_(value)

        # Create empty shared tensor
        shared = SharedTensor.empty((50, 50), torch.float32)
        shared.tensor.zero_()

        # Write from child process
        handle = shared.get_handle()

        p = Process(target=writer_process, args=(handle, 42.0))
        p.start()
        p.join()

        # Verify in main process
        assert torch.all(shared.tensor == 42.0)

        shared.drop()

    def test_multiprocess_bidirectional(self):
        """Test bidirectional communication"""

        def worker_process(input_handle, output_handle):
            input_tensor = SharedTensor(handle=input_handle).tensor
            output_tensor = SharedTensor(handle=output_handle).tensor

            # Compute: output = input * 2
            output_tensor.copy_(input_tensor * 2)

        # Create input and output tensors
        input_shared = SharedTensor.empty((100, 100), torch.float32)
        input_shared.tensor.normal_(0, 1)
        input_data = input_shared.tensor.clone()

        output_shared = SharedTensor.empty((100, 100), torch.float32)

        # Process in child
        p = Process(
            target=worker_process,
            args=(input_shared.get_handle(), output_shared.get_handle()),
        )
        p.start()
        p.join()

        # Verify result
        expected = input_data * 2
        assert torch.allclose(
            output_shared.tensor, expected
        ), "output: {}, expected: {}".format(output_shared.tensor, expected)

        input_shared.drop()
        output_shared.drop()


class TestSharedTensorPerformance:
    """Performance-related tests"""

    def test_empty_faster_than_from_tensor(self):
        """Test that empty() is faster than from tensor"""
        shape = (1000, 1000)

        # Time empty creation
        start = time.time()
        for _ in range(10):
            shared = SharedTensor.empty(shape, torch.float32)
            shared.drop()
        empty_time = time.time() - start

        # Time from_tensor creation
        start = time.time()
        for _ in range(10):
            tensor = torch.randn(shape)
            shared = SharedTensor(tensor=tensor)
            shared.drop()
        from_tensor_time = time.time() - start

        # empty() should be faster (no data copying)
        assert empty_time < from_tensor_time

    def test_handle_serialization_fast(self):
        """Test that handle serialization is fast"""
        shared = SharedTensor.empty((10000, 10000), torch.float32)
        handle = shared.get_handle()

        start = time.time()
        for _ in range(1000):
            pickled = pickle.dumps(handle)
            unpickled = pickle.loads(pickled)
        elapsed = time.time() - start

        # Should be able to do 1000 round trips in < 0.1 seconds
        assert elapsed < 0.1

        shared.drop()


class TestSharedTensorHandleToSharedTensor:
    """Test SharedTensorHandle.to_shared_tensor() method"""

    def test_to_shared_tensor_basic(self):
        """Test basic creation of SharedTensor from handle using to_shared_tensor method"""
        original = SharedTensor.empty((10, 10), torch.float32)
        original.tensor.fill_(7.0)

        handle = original.get_handle()
        reconstructed = handle.to_shared_tensor()

        assert torch.all(reconstructed.tensor == 7.0)
        assert reconstructed.tensor.shape == original.tensor.shape
        assert reconstructed.tensor.dtype == original.tensor.dtype

        original.drop()

    def test_to_shared_tensor_preserves_data(self):
        """Test that to_shared_tensor preserves original data"""
        original = SharedTensor.empty((20, 30), torch.float32)
        original.tensor.normal_(0, 1)
        original_data = original.tensor.clone()

        handle = original.get_handle()
        reconstructed = handle.to_shared_tensor()

        assert torch.allclose(reconstructed.tensor, original_data)

        original.drop()

    def test_to_shared_tensor_shares_memory(self):
        """Test that to_shared_tensor shares memory with original"""
        original = SharedTensor.empty((15, 15), torch.float32)
        original.tensor.fill_(5.0)

        handle = original.get_handle()
        reconstructed = handle.to_shared_tensor()

        reconstructed.tensor.fill_(10.0)

        assert torch.all(original.tensor == 10.0)

        original.drop()

    def test_to_shared_tensor_with_various_dtypes(self):
        """Test to_shared_tensor works with different data types"""
        for dtype in [torch.float32, torch.float64, torch.bfloat16, torch.int32]:
            original = SharedTensor.empty((5, 5), dtype)
            if (
                dtype == torch.bfloat16
                or dtype == torch.float32
                or dtype == torch.float64
            ):
                original.tensor.normal_(0, 1)
            else:
                original.tensor.fill_(42)

            handle = original.get_handle()
            reconstructed = handle.to_shared_tensor()

            assert reconstructed.tensor.dtype == dtype
            if dtype == torch.bfloat16:
                assert torch.allclose(
                    reconstructed.tensor.float(), original.tensor.float(), rtol=1e-3
                )
            else:
                assert torch.allclose(reconstructed.tensor, original.tensor)

            original.drop()

    def test_to_shared_tensor_multiprocess(self):
        """Test to_shared_tensor in multiprocess scenario"""

        def worker_process(handle, result_queue):
            shared = handle.to_shared_tensor()
            result_queue.put(shared.tensor.sum().item())

        original = SharedTensor.empty((50, 50), torch.float32)
        original.tensor.fill_(3.0)

        handle = original.get_handle()
        result_queue = Queue()

        p = Process(target=worker_process, args=(handle, result_queue))
        p.start()
        p.join()

        result = result_queue.get()
        expected = 3.0 * 50 * 50

        assert abs(result - expected) < 1e-5

        original.drop()

    def test_to_shared_tensor_equivalent_to_constructor(self):
        """Test that handle.to_shared_tensor() is equivalent to SharedTensor(handle=handle)"""
        original = SharedTensor.empty((25, 25), torch.float32)
        original.tensor.normal_(0, 1)

        handle = original.get_handle()

        via_method = handle.to_shared_tensor()
        via_constructor = SharedTensor(handle=handle)

        assert torch.allclose(via_method.tensor, via_constructor.tensor)
        assert via_method.tensor.shape == via_constructor.tensor.shape
        assert via_method.tensor.dtype == via_constructor.tensor.dtype

        original.drop()


class TestSharedTensorBfloat16:
    """Specific tests for bfloat16 support"""

    def test_bfloat16_creation(self):
        """Test bfloat16 tensor creation"""
        shared = SharedTensor.empty((100, 100), torch.bfloat16)
        assert shared.tensor.dtype == torch.bfloat16
        shared.drop()

    def test_bfloat16_from_tensor(self):
        """Test creating shared tensor from bfloat16 tensor"""
        original = torch.randn(50, 50, dtype=torch.bfloat16)
        shared = SharedTensor(tensor=original)

        assert shared.tensor.dtype == torch.bfloat16
        assert torch.allclose(shared.tensor.float(), original.float(), rtol=1e-3)

        shared.drop()

    def test_bfloat16_handle_preservation(self):
        """Test bfloat16 dtype preserved through handle"""
        shared1 = SharedTensor.empty((20, 20), torch.bfloat16)
        shared1.tensor.normal_(0, 1)

        handle = shared1.get_handle()
        shared2 = SharedTensor(handle=handle)

        assert shared2.tensor.dtype == torch.bfloat16
        assert torch.allclose(shared1.tensor.float(), shared2.tensor.float(), rtol=1e-3)

        shared1.drop()

    def test_bfloat16_operations(self):
        """Test operations on bfloat16 tensors"""
        shared = SharedTensor.empty((100, 100), torch.bfloat16)
        tensor = shared.tensor

        tensor.normal_(0, 1)
        mean = tensor.float().mean().item()

        # Mean should be close to 0
        assert abs(mean) < 0.1

        shared.drop()


class TestSharedTensorCloseAndCleanup:
    """Test explicit close() and cleanup patterns to prevent memory leaks"""

    def test_close_method(self):
        """Test explicit close() releases handle and sets closed state"""
        shared = SharedTensor.empty((10, 10), torch.float32)
        shared.tensor.fill_(5.0)

        assert not shared.is_closed

        # Close should not raise
        shared.close()

        assert shared.is_closed

        # Cleanup
        shared._shm.unlink()

    def test_tensor_access_after_close_raises_error(self):
        """Test that accessing tensor after close raises RuntimeError"""
        shared = SharedTensor.empty((10, 10), torch.float32)
        shared.tensor.fill_(5.0)

        shared.close()

        with pytest.raises(RuntimeError, match="Cannot access tensor after close"):
            _ = shared.tensor

        # Cleanup
        shared._shm.unlink()

    def test_get_handle_after_close_raises_error(self):
        """Test that getting handle after close raises RuntimeError"""
        shared = SharedTensor.empty((10, 10), torch.float32)

        shared.close()

        with pytest.raises(RuntimeError, match="Cannot get handle after close"):
            shared.get_handle()

        # Cleanup
        shared._shm.unlink()

    def test_is_closed_property(self):
        """Test is_closed property reflects state correctly"""
        shared = SharedTensor.empty((10, 10), torch.float32)

        assert not shared.is_closed

        shared.close()

        assert shared.is_closed

        # Cleanup
        shared._shm.unlink()

    def test_cached_tensor_reference_becomes_invalid_after_close(self):
        """Test that tensor reference obtained before close becomes invalid after close"""
        shared = SharedTensor.empty((10, 10), torch.float32)
        shared.tensor.fill_(5.0)

        # Get reference before close
        tensor_ref = shared.tensor

        shared.close()

        # After close(), the memory mapping is unmapped, so even cached references
        # point to invalid memory. Accessing them will cause segfault or undefined behavior.
        # We can't safely test this, but we document it.

        # Accessing via shared.tensor raises error (this is what we CAN test)
        with pytest.raises(RuntimeError):
            _ = shared.tensor

        # Cleanup
        shared._shm.unlink()

    def test_context_manager(self):
        """Test context manager automatically closes"""
        shm_name = None

        with SharedTensor.empty((10, 10), torch.float32) as shared:
            shm_name = shared._shm_name
            shared.tensor.fill_(7.0)
            assert torch.all(shared.tensor == 7.0)

        # After exiting context, should be closed (but not unlinked yet)
        # We need to unlink separately
        from multiprocessing import shared_memory

        # Should still be able to attach (not unlinked)
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.close()
        shm.unlink()

    def test_creator_receiver_workflow(self):
        """Test proper workflow: creator creates, gets handle, closes, receiver uses and closes"""

        def receiver_process(handle, result_queue):
            # Receiver creates SharedTensor from handle
            with SharedTensor(handle=handle) as shared:
                result = shared.tensor.sum().item()
                result_queue.put(result)
            # Context manager auto-closes

        # Creator process
        shared = SharedTensor.empty((50, 50), torch.float32)
        shared.tensor.fill_(4.0)
        handle = shared.get_handle()
        shared.close()  # Creator closes its reference

        # Pass to receiver
        result_queue = Queue()
        p = Process(target=receiver_process, args=(handle, result_queue))
        p.start()
        p.join()

        result = result_queue.get()
        assert abs(result - (4.0 * 50 * 50)) < 1e-5

        # Unlink after all processes done
        handle.drop()

    def test_handle_drop_without_creating_shared_tensor(self):
        """Test that handle.drop() doesn't create unnecessary SharedTensor instance"""
        shared = SharedTensor.empty((10, 10), torch.float32)
        shared.tensor.fill_(3.0)
        handle = shared.get_handle()
        shared.close()

        # drop() should work without creating new SharedTensor
        handle.drop()

        # Memory should be unlinked
        from multiprocessing import shared_memory

        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)

    def test_multiple_receivers_close_independently(self):
        """Test that multiple receivers can close independently"""

        def receiver_process(handle, value, result_queue):
            shared = SharedTensor(handle=handle)
            result = shared.tensor[0, 0].item() == value
            shared.close()  # Each receiver closes its own reference
            result_queue.put(result)

        # Creator
        shared = SharedTensor.empty((10, 10), torch.float32)
        shared.tensor.fill_(9.0)
        handle = shared.get_handle()
        shared.close()

        # Multiple receivers
        result_queue = Queue()
        processes = []
        for _ in range(3):
            p = Process(target=receiver_process, args=(handle, 9.0, result_queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # All should succeed
        for _ in range(3):
            assert result_queue.get() is True

        # Cleanup
        handle.drop()

    def test_close_is_idempotent(self):
        """Test that calling close() multiple times is safe"""
        shared = SharedTensor.empty((10, 10), torch.float32)

        # Multiple closes should not raise
        shared.close()
        shared.close()
        shared.close()

        # Cleanup
        shared.drop()

    def test_drop_is_idempotent(self):
        """Test that calling drop() multiple times is safe"""
        shared = SharedTensor.empty((10, 10), torch.float32)
        handle = shared.get_handle()
        shared.close()

        # Multiple drops should not raise
        handle.drop()
        handle.drop()
        handle.drop()

    def test_proper_cleanup_prevents_leak(self):
        """Test that proper close + unlink pattern doesn't leak"""
        import glob

        # Get initial shared memory count
        shm_before = len(glob.glob("/dev/shm/shared_tensor_*"))

        # Create and properly cleanup 10 shared tensors
        for _ in range(10):
            shared = SharedTensor.empty((100, 100), torch.float32)
            handle = shared.get_handle()
            shared.close()
            handle.drop()

        # Check no leaks
        shm_after = len(glob.glob("/dev/shm/shared_tensor_*"))
        assert (
            shm_after == shm_before
        ), f"Memory leak detected: {shm_after - shm_before} tensors leaked"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
