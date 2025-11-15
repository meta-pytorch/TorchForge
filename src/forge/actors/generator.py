# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torchstore as ts

from forge.actors._torchstore_utils import (
    extract_param_name,
    get_dcp_whole_state_dict_key,
    get_param_key,
    get_param_prefix,
    load_tensor_from_dcp,
    rdma_available,
)

from forge.controller import (
    ForgeActor,
    get_proc_mesh,
    host_mesh_from_proc,
    stop_proc_mesh,
)
from forge.data_models.completion import Completion
from forge.data_models.prompt import to_prompt
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.types import ProcessConfig
from forge.util._shared_tensor import SharedTensor, SharedTensorHandle
from monarch.actor import current_rank, endpoint, ProcMesh, this_host

from vllm.config import VllmConfig

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.executor.multiproc_worker_utils import set_multiprocessing_worker_envs
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_distributed_init_method
from vllm.v1.core.kv_cache_utils import get_kv_cache_config
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.worker.worker_base import WorkerWrapperBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Generator(ForgeActor):
    """Instance of a vLLM-based generator.

    This class manually recreates a vLLM engine that mirrors the design of AsyncLLMEngine in v1. The
    main difference is that all communications are controlled here via Monarch's proc meshes.

    Args:
        engine_args (EngineArgs): The engine arguments to use for the vLLM engine.
        sampling_params (SamplingParams): The sampling parameters to use for the vLLM engine.
        use_dcp_for_weight_sync (bool): Whether to use DCP for NFS-based weight sync. Default depends on
            whether or not RDMA is enabled in torchstore. If it is, then DCP is disabled. Otherwise, DCP is enabled.

    Example:
    >>> generator = await Generator.options(procs=1, num_replicas=1, with_gpus=True).as_service(
    ...     engine_args=EngineArgs(...),
    ...     sampling_params=SamplingParams(...),
    ...     )
    >>> await generator.generate("Tell me a joke")
    Completion(prompt="Tell me a joke", text="A: Why did the chicken cross the road? B: To get to the other side.",
    token_ids=[...], logprobs=[...])
    >>> await generator.shutdown()
    """

    engine_args: EngineArgs | Mapping = field(default_factory=EngineArgs)
    sampling_params: SamplingParams | Mapping = field(default_factory=SamplingParams)
    use_dcp_for_weight_sync: bool | None = None
    prefetch_weights_to_shm: bool = True
    n_fetcher_procs: int = 8

    def __post_init__(self):
        super().__init__()
        self._run_task: asyncio.Task | None = None
        self._generator_proc: ProcMesh | None = None
        self._worker_procs: ProcMesh | None = None
        self.worker: GeneratorWorker | None = None
        self.running = False
        self.generator_version: int = 0

        if isinstance(self.engine_args, Mapping):
            self.engine_args = EngineArgs(**self.engine_args)
        self.engine_args._is_v1_supported_oracle = lambda *_: True
        self.vllm_config = self.engine_args.create_engine_config(UsageContext.LLM_CLASS)

        if isinstance(self.sampling_params, Mapping):
            self.sampling_params = SamplingParams.from_optional(**self.sampling_params)
            self.sampling_params.output_kind = RequestOutputKind.FINAL_ONLY

        if self.use_dcp_for_weight_sync is None:
            self.use_dcp_for_weight_sync = not rdma_available()
        logger.debug(f"{self.use_dcp_for_weight_sync=}")

    @endpoint
    async def get_vllm_config(self) -> VllmConfig:
        return self.vllm_config

    @endpoint
    async def register_worker(self, worker: GeneratorWorker) -> None:
        self.worker = worker
        logger.debug("Registered GeneratorWorker on Generator.")

    @classmethod
    async def launch(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type["Generator"],
        *args,
        **kwargs,
    ) -> "Generator":
        """Custom launch for the Generator service with its workers.

        We overwrite the default Service launch method in order to setup Actors (GeneratorWorker) within this "coordinating" Actor.
        We first create a proc_mesh for the workers, then a proc_mesh for the generator, and then we spawn the workers
        and the generator in setup.
        """
        # Note: get_proc_mesh will set MASTER_ADDR, MASTER_PORT and CUDA_VISIBLE_DEVICES
        process_config: ProcessConfig = ProcessConfig(
            procs=cls.procs,
            hosts=cls.hosts,
            with_gpus=cls.with_gpus,
            mesh_name=cls.mesh_name,
        )

        # First, spawn the worker processes which may or may not be
        # on remote hosts.
        worker_procs = await get_proc_mesh(process_config=process_config)

        # Then, grab a single host from the workers...
        host_mesh = await host_mesh_from_proc(worker_procs)
        singleton_slice = {k: slice(0, 1) for k in host_mesh.extent.keys()}
        host_mesh = host_mesh.slice(**singleton_slice)

        # We ask the provisioner for a single process on a single host
        generator_proc_config = copy(process_config)
        generator_proc_config.procs = 1
        generator_proc_config.with_gpus = False
        generator_proc = await get_proc_mesh(
            process_config=generator_proc_config, host_mesh=host_mesh
        )
        # TODO - expand support so name can stick within kwargs
        actor_name = kwargs.pop("name", cls.__name__)
        generator = generator_proc.spawn(
            actor_name,
            cls,
            *args,
            **kwargs,
        )

        vllm_config = (
            await generator.get_vllm_config.call_one()
        )  # Config should be the same across all actors
        worker = worker_procs.spawn(
            "vllm_worker", GeneratorWorker, vllm_config=vllm_config
        )
        await worker.setup.call()
        await generator.register_worker.call(worker)

        generator._generator_proc = generator_proc
        generator._worker_procs = worker_procs
        await generator.setup.call()

        return generator

    @endpoint
    async def setup(self):
        """Mirrors the __init__ of vLLM's LLMEngine."""
        self.request_id = 0
        self.requests: dict[str, tuple[ParentRequest | None, asyncio.Future]] = {}

        # TODO: Investigate whether this can be combined with `generator.running`
        self.accepting_requests = True

        self.request_lock = asyncio.Condition()  # Guard for accepting_requests
        self.update_lock = asyncio.Condition()  # Guard for updating requests

        # Setup processors
        # TODO: move all processing to the Environment
        # TODO: add support for `log_stats` and `mm_registry`
        tokenizer = init_tokenizer_from_configs(
            model_config=self.vllm_config.model_config,
            scheduler_config=self.vllm_config.scheduler_config,
            lora_config=self.vllm_config.lora_config,
        )
        self.processor = Processor(
            vllm_config=self.vllm_config, tokenizer=tokenizer, mm_registry=None
        )
        self.output_processor = OutputProcessor(tokenizer, log_stats=None)

        # Configure KV caches
        kv_cache_configs = await self.worker.setup_kv_cache.call()
        _, kv_cache_config = next(kv_cache_configs.items())
        self.vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.vllm_config.cache_config.num_cpu_blocks = 0

        # Setup scheduler
        # TODO: Add support for `log_stats`
        structured_output_manager = StructuredOutputManager(self.vllm_config)
        self.scheduler = Scheduler(
            vllm_config=self.vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            include_finished_set=False,
            log_stats=None,
        )
        self._start_processing()
        if self.prefetch_weights_to_shm:
            self._spawn_fetchers()

    def _spawn_fetchers(self):
        """Spawn weight fetchers that prefetch weights from torchstore to shared memory."""
        # TODO: this assumes the generator is on the same host as the worker
        # and only works for single host generators. Figure out how to support
        # generators with workers spanned across multiple hosts.
        fetcher_procs = this_host().spawn_procs(
            per_host={"procs": self.n_fetcher_procs}
        )
        self._fetcher_procs = fetcher_procs
        self.weight_fetchers = fetcher_procs.spawn("weight_fetcher", _WeightFetcher)

    def _start_processing(self):
        if self._run_task is None or self._run_task.done():
            self._run_task = asyncio.create_task(self.run())

    async def _drop_shared_memory(self, state_dict: dict[str, SharedTensorHandle]):
        for handle in state_dict.values():
            handle.drop()

    async def _fetch_weights(
        self,
        version: int,
    ) -> dict[str, SharedTensorHandle]:
        """Fetch weights from torchstore and return a dict of {name: SharedTensorHandle}."""
        t = Tracer("generator_perf/_fetch_weights")
        t.start()
        prefix = get_param_prefix(version)
        matching_keys = await ts.keys(prefix)
        hf_param_names = [extract_param_name(key) for key in matching_keys]

        n_fetchers = self.weight_fetchers.size()

        def split_keys(keys):
            return [keys[i::n_fetchers] for i in range(n_fetchers)]

        futures = []
        for i, names in enumerate(split_keys(hf_param_names)):
            fut = self.weight_fetchers.slice(procs=i).fetch.call_one(
                version=version, param_names=names
            )
            futures.append(fut)

        sub_state_dicts = [await fut for fut in futures]

        state_dict = {}
        for sd in sub_state_dicts:
            state_dict.update(sd)

        t.stop()

        return state_dict

    @endpoint
    async def generate(
        self,
        prompt: str,
        *,
        priority: int = 0,
        sampling_params: SamplingParams | None = None,
    ) -> list[Completion]:
        """Generate a response for the given prompt

        Args:
            prompt (str): The prompt to generate a response for.
            priority (int, optional): The priority of the request. Defaults to 0.
            sampling_params (SamplingParams, optional): Sampling parameters to use for this request.
                If not provided, uses self.sampling_params.

        Returns:
            list[Completion]: n completions from vLLM based on your prompt.
        """
        t = Tracer("generator_perf/generate", timer="gpu")
        t.start()
        record_metric("generator/generate/count_requests", 1, Reduce.SUM)

        if sampling_params is not None:
            # as in `post_init`
            sampling_params.output_kind = RequestOutputKind.FINAL_ONLY

        params = sampling_params or self.sampling_params

        self.request_id += 1 % sys.maxsize
        request_id = str(self.request_id)

        tokenization_kwargs = {}
        # TODO: add truncation support https://github.com/vllm-project/vllm/issues/4507
        truncate_prompt_tokens = params.truncate_prompt_tokens
        _validate_truncation_size(
            self.vllm_config.model_config.max_model_len,
            truncate_prompt_tokens,
            tokenization_kwargs,
        )
        prompt_str, request = self.processor.process_inputs(
            request_id=request_id,
            prompt={"prompt": prompt},
            params=params,
            arrival_time=None,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=None,
            priority=priority,
            data_parallel_rank=None,  # We do not support DP
        )
        t.step("process_inputs")

        # Wait until we're accepting requests (releases lock while waiting)
        # If accepting_requests is True, continue immediately (holding the lock)
        # If False, release lock, wait for notification, re-acquire and recheck
        async with self.request_lock:
            await self.request_lock.wait_for(lambda: self.accepting_requests)

            # Explicitly keeping the redundant logic to make it easier to pick up vLLM changes
            if (num_samples := params.n) == 1:
                self.output_processor.add_request(request, prompt_str, None, 0)
                request, _ = self._preprocess_add_request(request)
                request_fut = asyncio.Future()
                self.requests[request_id] = (None, request_fut)
                self.scheduler.add_request(request)
            else:
                parent_req = ParentRequest(request_id, params)
                for idx in range(num_samples):
                    # Note: `get_child_info` mutates ParentRequest to track the
                    # generated child request
                    child_request_id, params_child = parent_req.get_child_info(idx)
                    child_request = request if idx == num_samples - 1 else copy(request)
                    child_request.request_id = child_request_id
                    child_request.sampling_params = params_child
                    self.output_processor.add_request(
                        child_request, prompt_str, parent_req, idx
                    )
                    child_request, _ = self._preprocess_add_request(child_request)
                    self.scheduler.add_request(child_request)
                request_fut = asyncio.Future()
                self.requests[request_id] = (parent_req, request_fut)

        completions = await request_fut
        t.step("generate")

        # Log some metrics
        record_metric(
            "generator/generate/count_sequences_completed",
            len(completions),
            Reduce.SUM,
        )

        for completion in completions:
            num_generated_tokens = len(completion.token_ids)
            record_metric(
                "generator/generate/sum_tokens_generated",
                num_generated_tokens,
                Reduce.SUM,
            )

            record_metric(
                "generator/generate/avg_tokens_generated",
                num_generated_tokens,
                Reduce.MEAN,
            )
        t.stop()
        return completions

    def _preprocess_add_request(
        self, request: EngineCoreRequest
    ) -> tuple[Request, int]:
        """(forge/issues/332) Will require attention when we bump vllm versions
        https://github.com/vllm-project/vllm/blob/0e3bb543f064eb416bca4f6f3013efa3830b12f7/vllm/v1/engine/core.py#L419
        """
        if request.mm_hashes is not None:
            raise NotImplementedError("Support for mm_hash is not implemented yet.")
        req = Request.from_engine_core_request(request)
        if req.use_structured_output:
            self.scheduler.structured_output_manager.grammar_init(request)
        return req, 0

    async def run(self) -> None:
        """Schedule, execute, and make output.
        https://github.com/vllm-project/vllm/blob/0e3bb543f064eb416bca4f6f3013efa3830b12f7/vllm/v1/engine/core.py#L276
        """
        # TODO: move postprocessing out of loop to not block
        self.running = True
        while self.running:
            scheduler_output = self.scheduler.schedule()
            worker_outputs = await self.worker.execute_model.call(scheduler_output)

            # The results of `execute_model` are gathered on the driver rank (rank 0)
            _, worker_output = next(worker_outputs.items())
            outputs = self.scheduler.update_from_output(scheduler_output, worker_output)
            outputs = outputs.get(0) or EngineCoreOutputs()
            await asyncio.sleep(0)  # Release control before processing outputs

            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=None,  # TODO: add support for `iteration_stats`
            )
            for request_output in processed_outputs.request_outputs:
                if request_output.finished:
                    completions = self._to_completions(request_output)
                    _, fut = self.requests.pop(request_output.request_id)
                    fut.set_result(completions)

            # Notify waiters if queue is drained
            async with self.request_lock:
                if len(self.requests) == 0:
                    self.request_lock.notify_all()

    @endpoint
    async def update_weights(self, version: int) -> None:
        """Update weights on base model from a generator version to be found in a torchstore volume.

        Args:
            generator_version (int): Generator version from which to update. This will correspond to a key in a
                torchstore volume.

        Example:
            >>> trainer.train_step(...)
            >>> version += 1
            >>> await trainer.push_weights()
            >>> generator.update_weights(version)
        """
        # TODO: enable shared memory prefetch for DCP-based weight sync
        if self.prefetch_weights_to_shm and not self.use_dcp_for_weight_sync:
            logger.info(f"[Generator] Fetching weights for v{version} to shared memory")
            fetch_fut = asyncio.create_task(self._fetch_weights(version))
        else:
            fetch_fut = None
        # Serialize updates (only one update at a time)
        async with self.update_lock:
            # Grab the lock to stop accepting requests and wait on pending requests
            async with self.request_lock:
                self.accepting_requests = False
                curr_requests = [fut for _, fut in self.requests.values()]
                if curr_requests:
                    # Record pending requests metrics
                    record_metric(
                        "generator_perf/update_weights/avg_pending_requests",
                        len(curr_requests),
                        Reduce.MEAN,
                    )
                    record_metric(
                        "generator_perf/update_weights/max_pending_requests",
                        len(curr_requests),
                        Reduce.MAX,
                    )
                    logger.debug(f"Waiting for {len(curr_requests)} pending requests")

                # Wait until all pending requests have been processed
                # TODO: If generating long sequences, this might be long and will block
                # generator weight updates
                await self.request_lock.wait_for(lambda: len(self.requests) == 0)

            # Record weight update metrics
            record_metric(
                "generator/update_weights/count_weight_updates", 1, Reduce.SUM
            )

            logger.debug(f"Starting weight update on {self.__class__.__name__}")

            if fetch_fut is not None:
                t = Tracer("generator_perf/waiting_for_fetch_weights")
                t.start()
                fetched_weights = await fetch_fut
                t.stop()
                # Call update_weights on every policy_worker
                await self.worker.update_weights.call(
                    shared_memory_state_dict=fetched_weights
                )
                await self._drop_shared_memory(fetched_weights)
            else:
                await self.worker.update_weights.call(version=version)
            self.generator_version = version

            # After updating the weights, we need to reset the KV cache
            self.scheduler.reset_prefix_cache()

        # Resume accepting requests and wake up any waiting generate() calls
        async with self.request_lock:
            self.accepting_requests = True
            self.request_lock.notify_all()

        logger.info(f"Weight update completed (now v{self.generator_version})")

    @endpoint
    async def _reset_prefix_cache(self):
        self.scheduler.reset_prefix_cache()

    @endpoint
    async def get_version(self) -> int:
        """Get the current generator version."""
        return self.generator_version

    @endpoint
    async def stop(self):
        self.running = False

    def _to_completions(self, request_output: RequestOutput) -> list[Completion]:
        """Convert a vLLM RequestOutput to a list of Completion objects."""
        completions = []
        original_prompt = request_output.prompt
        prompt_token_ids = request_output.prompt_token_ids
        for output in request_output.outputs:
            completions.append(
                Completion(
                    # TODO: the to_prompt encoding will be different from the original.
                    # This is okay for now, since I don't see any direct usage of prompt using completion object.
                    prompt=to_prompt(original_prompt),
                    stop_reason=output.finish_reason,
                    text=output.text,
                    prompt_ids=torch.tensor(prompt_token_ids),
                    token_ids=torch.tensor(output.token_ids),
                    logprobs=self._extract_logprobs(output),
                    generator_version=self.generator_version,
                    metadata={"num_cached_tokens": request_output.num_cached_tokens},
                )
            )
        return completions

    def _extract_logprobs(self, sample: CompletionOutput) -> torch.Tensor | None:
        if sample.logprobs is not None:
            return torch.tensor(
                [
                    top_k_dict[token].logprob
                    for token, top_k_dict in zip(sample.token_ids, sample.logprobs)
                ]
            )
        return None

    @classmethod
    async def shutdown(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type["Generator"], actor: "Generator"
    ):
        assert (
            actor._generator_proc is not None
        ), "Tried to shutdown a generator that was not initialized correctly"
        assert (
            actor._worker_procs is not None
        ), "Tried to shutdown a generator that was not initialized correctly"

        # TODO - may want to expand stop to gracefully respond to
        # ongoing requests.
        await actor.stop.call()
        await stop_proc_mesh(actor._worker_procs)
        await stop_proc_mesh(actor._generator_proc)
        await stop_proc_mesh(actor._fetcher_procs)

    @endpoint
    async def save_model_params(self):
        """Save model parameters before weight update, used for testing purposes only."""
        logger.info("[Generator] save model parameters for testing.")
        await self.worker.save_model_params.call()

    @endpoint
    async def validate_model_params(self, validate_fn):
        """Validate updated model params using validate_fn."""
        logger.info("[Generator] start validating model parameters.")
        return await self.worker.validate_model_params.call(validate_fn)


@dataclass
class GeneratorWorker(ForgeActor):
    """Mirrors a vLLM GPUWorker
    https://github.com/vllm-project/vllm/blob/0e3bb543f064eb416bca4f6f3013efa3830b12f7/vllm/v1/worker/gpu_worker.py

    In general, this class should not be instantiated or called directly. Rather, the Generator controls
    the creation and invocation of all GeneratorWorker.
    """

    vllm_config: VllmConfig
    # TODO: Remove below param
    _test_prev_params = {}

    def __post_init__(self):
        super().__init__()

    @endpoint
    async def setup(self):
        self.rank = current_rank().rank
        os.environ["RANK"] = str(self.rank)
        parallel_config = self.vllm_config.parallel_config
        set_multiprocessing_worker_envs(parallel_config)
        ip, port = os.getenv("MASTER_ADDR"), os.getenv("MASTER_PORT")
        distributed_init_method = get_distributed_init_method(ip, port)
        all_kwargs = [{}] * parallel_config.world_size
        local_rank = self.rank % torch.accelerator.device_count()
        is_driver_worker = self.rank % parallel_config.tensor_parallel_size == 0
        all_kwargs[self.rank] = {
            "vllm_config": self.vllm_config,
            "local_rank": local_rank,
            "rank": self.rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
        }
        self.worker = WorkerWrapperBase(self.vllm_config, self.rank)
        self.worker.init_worker(all_kwargs)
        self.worker.init_device()
        self.worker.load_model()

    @endpoint
    async def setup_kv_cache(self) -> KVCacheConfig:
        """https://github.com/vllm-project/vllm/blob/5c7fe25491825b95936c011a43337c7d4fb7e472/vllm/v1/engine/core.py#L199"""
        kv_cache_spec = self.worker.get_kv_cache_spec()
        if kv_cache_spec is not None:
            available_gpu_memory = self.worker.determine_available_memory()
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = 0

        # Get the kv cache tensor size
        kv_cache_config = get_kv_cache_config(
            self.vllm_config, kv_cache_spec, available_gpu_memory
        )
        # TODO: unify configs across TorchStore
        # unify_kv_cache_configs(kv_cache_configs)
        self.vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.vllm_config.cache_config.num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution:
        # from multiproc_executor.py:MultiprocExecutor.initialize_from_config
        kv_cache_configs = [None] * self.vllm_config.parallel_config.world_size
        kv_cache_configs[self.rank] = kv_cache_config
        self.worker.initialize_from_config(kv_cache_configs)
        self.worker.compile_or_warm_up_model()
        self.worker.initialize_cache(kv_cache_config.num_blocks, 0)
        return kv_cache_config

    @endpoint
    async def execute_model(self, schedule: SchedulerOutput) -> ModelRunnerOutput:
        return self.worker.execute_model(schedule)

    @endpoint
    async def update_weights(
        self,
        version: Optional[int] = None,
        *,
        shared_memory_state_dict: Optional[dict[str, SharedTensorHandle]] = None,
    ) -> None:
        model = self.worker.model_runner.model
        if shared_memory_state_dict is not None:
            logger.info("[PolicyWorker] update weights from shared memory.")
            t = Tracer(
                "generator_worker_perf/update_weights_from_shared_memory", timer="gpu"
            )
            t.start()
            loaded_weights = set()
            for name, param_handle in shared_memory_state_dict.items():
                # Use context manager for automatic cleanup
                with param_handle.to_shared_tensor() as shared_tensor:
                    param = shared_tensor.tensor
                    loaded = model.load_weights([(name, param)])
                    del param
                    loaded_weights.update(loaded)
            logger.info(f"[PolicyWorker] updated {len(loaded_weights)} parameters")
            t.stop()
            return
        # normal update_weights without shared memory prefetching
        if version is None:
            raise ValueError(
                "version must be provided if not using shared_memory_state_dict"
            )
        logger.info("[PolicyWorker] update weights from torchstore.")
        prefix = get_param_prefix(version)
        matching_keys = await ts.keys(prefix)
        dcp_whole_state_dict_key = get_dcp_whole_state_dict_key(version)
        use_dcp_for_weight_sync = dcp_whole_state_dict_key in matching_keys
        loaded_weights = set()
        t = Tracer("generator_worker_perf/update_weights_from_torchstore", timer="gpu")
        t.start()

        if use_dcp_for_weight_sync:
            dcp_handle = await ts.get(dcp_whole_state_dict_key)
            hf_param_names = dcp_handle.param_names
            for name in hf_param_names:
                param = load_tensor_from_dcp(dcp_handle, name)
                loaded = model.load_weights([(name, param)])
                del param
                loaded_weights.update(loaded)
        else:
            hf_param_names = [extract_param_name(key) for key in matching_keys]
            # We can't pass a generator since vllm load_weights is not async.
            # Instead, we just call load_weights with one parameter at a time.
            for name in hf_param_names:
                param_key = get_param_key(version, name)
                param = await ts.get(param_key)
                loaded = model.load_weights([(name, param)])
                del param
                loaded_weights.update(loaded)

        t.stop()

    @endpoint
    async def save_model_params(self):
        """Save model parameters before weight update, used for testing purposes only."""
        logger.info("[GeneratorWorker] save model parameters for testing.")
        for name, param in self.worker.model_runner.model.named_parameters():
            self._test_prev_params[name] = param.detach().cpu()
        logger.info(
            "[GeneratorWorker] finished saving model parameters, len = %d",
            len(self._test_prev_params),
        )

    @endpoint
    async def validate_model_params(self, validate_fn):
        """Validate updated model params using validate_fn."""
        logger.info("[GeneratorWorker] start validating model parameters.")
        return validate_fn(
            self._test_prev_params, self.worker.model_runner.model, logger
        )


class _WeightFetcher(ForgeActor):
    """Fetches weights from torchstore and loads them into shared memory.
    This has to be colocated with the GeneratorWorker."""

    @endpoint
    async def fetch(
        self,
        *,
        version: int,
        param_names: list[str],
    ) -> dict[str, SharedTensorHandle]:
        """Fetch weights from torchstore and load them into shared memory."""
        sd = {}
        for name in param_names:
            param_key = get_param_key(version, name)
            param = await ts.get(param_key)
            # Use context manager to ensure cleanup after getting handle
            with SharedTensor(tensor=param) as shared_tensor:
                handle = shared_tensor.get_handle()
                sd[name] = handle
            del param  # Explicitly free the tensor after copying to shared memory
        return sd
