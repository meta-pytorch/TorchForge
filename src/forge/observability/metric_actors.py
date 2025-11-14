# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import uuid
from typing import Any, Union

from monarch.actor import (
    context,
    endpoint,
    get_or_spawn_controller,
    ProcMesh,
    this_proc,
)

from forge.controller.actor import ForgeActor

from forge.env import FORGE_DISABLE_METRICS
from forge.observability.metrics import (
    BackendRole,
    get_logger_backend_class,
    LoggerBackend,
    LoggingMode,
    MetricCollector,
    Reduce,
    reduce_metrics_states,
)


logger = logging.getLogger(__name__)

_global_logger = None


async def get_or_create_metric_logger(
    proc_mesh: ProcMesh | None = None,
    process_name: str | None = None,
) -> "GlobalLoggingActor":
    """Spawns a LocalFetcherActor for the specified ProcMesh (if not already initialized),
    registers it with the GlobalLoggingActor, and returns the GlobalLoggingActor.

    Usage:
    1. Main process: call `get_or_create_metric_logger()` to get the global logger
    2. Service spawning: call `get_or_create_metric_logger(proc_mesh, process_name)` to register the
        map(proc_mesh,local fetcher) with the global logger, so it knows to broadcast to all ranks.

    Args:
        proc_mesh: Optional ProcMesh to spawn LocalFetcherActor on. If None, uses `this_proc()`.
        process_name: Optional process name (e.g., "TrainActor") for logging. Auto-detected from the context if None.

    Returns:
        GlobalLoggingActor: The global logging controller.

    Raises:
        ValueError: If the logging state is inconsistent.

    Example:
        from forge.observability.metric_actors import get_or_create_metric_logger
        from forge.observability.metrics import record_metric

        # Main process setup
        mlogger = await get_or_create_metric_logger(process_name="Controller")

        # Initialize logging backends
        await mlogger.init_backends({
            "console": {"logging_mode": "global_reduce"},
            "wandb": {"project": "my_project", "logging_mode": "per_rank_reduce"}
        })

        # Initialize services...
        policy = await Generator.as_service(...)

        # Training loop
        for step in range(max_steps):
            record_metric("loss", 1.2, reduction_type=Reduce.MEAN)
            # ... training code with record_metric() calls ...
            await mlogger.flush.call_one(step)  # Log metrics for this step

        # Shutdown
        await mlogger.shutdown.call_one()
    """

    # Get or create the singleton global logger
    global _global_logger

    if _global_logger is None:
        _global_logger = await get_or_spawn_controller(
            "global_logger", GlobalLoggingActor
        )
    global_logger = _global_logger

    # Determine process context
    proc = proc_mesh if proc_mesh is not None else this_proc()

    # Auto-detect process_name from proc mesh if not provided
    if process_name is None:
        ctx = context()
        process_name = ctx.actor_instance.actor_id.actor_name

    # Check current state for consistency
    proc_has_local_fetcher = hasattr(proc, "_local_fetcher")
    proc_id = proc._uid if proc_has_local_fetcher else None
    global_logger_has_local_fetcher = await global_logger.has_fetcher.call_one(proc_id)

    # Consistency check: both should be in sync
    if proc_has_local_fetcher != global_logger_has_local_fetcher:
        raise ValueError(
            f"Inconsistent logging state for proc {proc}: "
            f"proc has _local_fetcher={proc_has_local_fetcher}, "
            f"but global_logger has registration={global_logger_has_local_fetcher}. "
            f"This indicates a bug in logging setup/teardown. "
            f"Both should be True (already setup) or both False (needs setup)."
        )

    # Setup local_fetcher_actor if needed (unless disabled by environment flag)
    if not proc_has_local_fetcher and not FORGE_DISABLE_METRICS.get_value():
        local_fetcher_actor = proc.spawn(
            "local_fetcher_actor", LocalFetcherActor, global_logger, process_name
        )
        # Generate a unique ID to map procmesh to fetcher
        proc._uid = str(uuid.uuid4())
        proc._local_fetcher = local_fetcher_actor  # pyre-ignore

        await global_logger.register_fetcher.call_one(local_fetcher_actor, proc._uid)

    return global_logger


class LocalFetcherActor(ForgeActor):
    """Actor spawned once per ProcMesh that, when called, runs on every rank in that ProcMesh
    and accesses each rank's local MetricCollector.

    Flow:
    GlobalLoggingActor.method() -> per-procmesh LocalFetcherActor.method() -> per-rank MetricCollector.method() -> logger
    """

    def __init__(
        self,
        global_logger: Union["GlobalLoggingActor", None] = None,
        process_name: str | None = None,
    ) -> None:
        self.global_logger = global_logger
        self.process_name = process_name

    @endpoint
    async def flush(
        self, global_step: int, return_state: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Log to local logger backends (if any), reset accumulators and return metric states dict if return_state=True.
        This should only ever be called by the global logger.

        Args:
            global_step (int): step used by backends to align all metrics on the same x-axis
            return_state (bool): Used by GlobalLoggingActor for reduction across all ranks.
                If False, returns empty dict, else returns the state of all metrics collected.
        Returns:
            dict[str, dict[str, Any]]: of {metric_key: metric_state},
                e.g., {"loss": {"reduction_type": "mean", "sum": 1.2, "count": 3}}.
        """
        collector = MetricCollector()
        result = await collector.flush(global_step, return_state=return_state)
        return result

    @endpoint
    async def init_backends(
        self,
        metadata_per_controller_backend: dict[str, dict[str, Any]],
        config: dict[str, Any],
        global_step: int = 0,
    ) -> None:
        """Init per-rank logger backends and MetricCollector.

        Args:
            metadata_per_controller_backend (dict[str, dict[str, Any]]): Metadata from controller backends for shared state.
            config (dict[str, Any]): Backend configurations with logging modes and settings.
            global_step (int): Initial step for metrics.
        """
        collector = MetricCollector()
        await collector.init_backends(
            metadata_per_controller_backend,
            config,
            global_step,
            process_name=self.process_name,
        )

    @endpoint
    async def shutdown(self) -> None:
        collector = MetricCollector()
        await collector.shutdown()


class GlobalLoggingActor(ForgeActor):
    """Coordinates metric logging across all ProcMeshes and their ranks.

    Supports multiple logging backends (e.g., WandB, TensorBoard, etc.),
    with per-rank and/or global reduction logging modes.

    This GlobalLoggingActor should be spawned once in the controller. A LocalFetcherActor
    is automatically spawned per-procmesh in `forge.controller.provisioner.py` and registered
    with this actor. The LocalFetcherActor is responsible for instantiating
    the per-rank MetricCollector and working as a bridge between GlobalLoggingActor and processes.

    Flow:
    GlobalLoggingActor.method() -> per-procmesh LocalFetcherActor.method() -> per-rank MetricCollector.method() -> logger
    """

    def __init__(self):
        self.fetchers: dict[str, LocalFetcherActor] = {}
        self.config: dict[str, Any] | None = None
        self.global_logger_backends: dict[str, LoggerBackend] = {}
        self.metadata_per_controller_backend: dict[str, dict[str, Any]] = {}

    def _validate_backend_config(
        self, backend_name: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate and normalize backend configuration."""
        if "logging_mode" not in config:
            raise ValueError(
                f"logging_mode is required for backend '{backend_name}' but was not provided. "
                f"Please specify a logging_mode in your config. "
                f"See forge.observability.metrics.LoggingMode for available options: "
                f"{', '.join([mode.value for mode in LoggingMode])}."
            )

        # Convert string to LoggingMode enum
        mode_value = config["logging_mode"]
        if isinstance(mode_value, str):
            mode = LoggingMode(mode_value)
        elif isinstance(mode_value, LoggingMode):
            mode = mode_value
        else:
            raise TypeError(
                f"logging_mode must be str or LoggingMode enum, got {type(mode_value)}"
            )

        # Validate per_rank_share_run configuration
        share_run = config.get("per_rank_share_run", False)
        if mode == LoggingMode.GLOBAL_REDUCE and share_run:
            logger.warning(
                f"{backend_name}: per_rank_share_run=True is ignored in {mode.value} mode. "
                "Setting it to False."
            )
            share_run = False

        # WandB-specific warning for suboptimal configuration
        if (
            backend_name == "wandb"
            and mode == LoggingMode.PER_RANK_REDUCE
            and share_run
        ):
            logger.warning(
                "WandB: Using 'per_rank_reduce' with 'per_rank_share_run=True' is not recommended. "
                "This configuration can lead to confusing metrics where reduced values from multiple ranks "
                "are written to the same run/step, displaying only one of them. Consider either:\n"
                "  1. Set 'per_rank_share_run=False' to create separate runs per rank, OR\n"
                "  2. Use 'per_rank_no_reduce' for real-time streaming to a shared run"
            )

        return {
            **config,
            "logging_mode": mode,
            "per_rank_share_run": share_run,
        }

    @endpoint
    async def init_backends(self, config: dict[str, Any]) -> None:
        """Sets config in global actor and initializes existing backends and collectors. Later spawned actors
        are initialized in `register_fetcher` endpoint.

        Controller backends (instantiated in the controller) can provide metadata to be shared with rank backends,
        e.g. shared run IDs for WandB. For details on logging modes, see `forge.observability.metrics.LoggingMode`.

        Args:
            config (dict[str, Any]): Config for metric logging where keys are backend names.
                Each backend config supports:
                - logging_mode (str | LoggingMode): Check LoggingMode for options. Defaults to "global_reduce".
                - per_rank_share_run (bool, default False): For per-rank modes only. Whether ranks
                  share a single run/logger instance. Ignored for "global_reduce" mode.
                - Additional backend-specific options (e.g., "project" for WandB)

                Example:
                {
                    "console": {"logging_mode": "global_reduce"},
                    "wandb": {
                        "logging_mode": "per_rank_no_reduce",
                        "per_rank_share_run": True,
                        "project": "my_project",
                    }
                }

        Raises:
            ValueError: If backend config is invalid or missing required fields.
        """
        self.config = {}

        # Skip initialization if disabled by environment flag
        if FORGE_DISABLE_METRICS.get_value():
            return

        # Validate and normalize each backend config
        for backend_name, backend_config in config.items():
            self.config[backend_name] = self._validate_backend_config(
                backend_name, backend_config
            )

        # Initialize backends based on logging mode
        for backend_name, backend_config in self.config.items():
            mode = backend_config["logging_mode"]

            backend: LoggerBackend = get_logger_backend_class(backend_name)(
                **backend_config
            )
            await backend.init(role=BackendRole.GLOBAL, process_name="global_reduce")

            # Extract metadata from controller logger to be shared with per-rank loggers
            if mode != LoggingMode.GLOBAL_REDUCE:
                controller_metadata: dict[str, Any] = (
                    backend.get_metadata_for_secondary_ranks() or {}
                )
                self.metadata_per_controller_backend[backend_name] = controller_metadata

            # Store global logger backends for later flush
            if mode == LoggingMode.GLOBAL_REDUCE:
                self.global_logger_backends[backend_name] = backend

        # Init collectors on all registered fetchers
        if self.fetchers:
            tasks = [
                fetcher.init_backends.call(
                    self.metadata_per_controller_backend, self.config
                )
                for fetcher in self.fetchers.values()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    @endpoint
    async def register_fetcher(self, fetcher: LocalFetcherActor, proc_id: str) -> None:
        """Registers a LocalFetcherActor with the GlobalLoggingActor. One LocalFetcherActor per ProcMesh.

        Args:
            fetcher: The LocalFetcherActor instance for a ProcMesh
            proc_id: Unique identifier for the ProcMesh
        """
        self.fetchers[proc_id] = fetcher

        # Self-init for respawned actors
        if self.config:
            logger.debug(f"Initializing new LocalFetcherActor for proc_id={proc_id}")
            await fetcher.init_backends.call(
                self.metadata_per_controller_backend, self.config
            )

    @endpoint
    async def deregister_fetcher(self, proc_id: str) -> None:
        if proc_id not in self.fetchers:
            logger.warning(
                f"Fetcher {proc_id} not registered in GlobalLoggingActor. Cannot deregister."
                f"Available fetchers: {self.fetchers.keys()}"
            )
            return
        del self.fetchers[proc_id]

    @endpoint
    async def flush(self, global_step: int) -> None:
        """
        Triggers parallel flush/reset on all registered fetchers. Per-rank MetricCollectors
        log to local backends and return states if needed for cross-rank reduction.

        Args:
            global_step (int): step for logging.
        """
        if not self.fetchers:
            return

        config = self.config
        if config is None:
            logger.warning(
                "Cannot flush collected metrics. GlobalLoggingActor.flush() called before init_backends()."
                " No backends will be flushed. Please call in your main file:\n"
                "`mlogger = await get_or_create_metric_logger(process_name='Controller')`\n"
                "`await mlogger.init_backends.call_one(logging_config)`\n"
            )
            return

        # Check if need to collect states from fetchers for global reduction
        needs_state_collection = any(
            backend_config["logging_mode"] == LoggingMode.GLOBAL_REDUCE
            for backend_config in config.values()
        )

        logger.debug(
            f"Global flush for global step {global_step}: {len(self.fetchers)} fetchers"
        )

        # Broadcast flush to all fetchers
        results = await asyncio.gather(
            *[
                f.flush.call(global_step, return_state=needs_state_collection)
                for f in self.fetchers.values()
            ],
            return_exceptions=True,
        )

        if needs_state_collection:

            def extract_values_from_valuemesh(results) -> list[dict[str, Any]]:
                all_local_states = []
                for result in results:
                    if isinstance(result, BaseException):
                        logger.warning(f"Flush failed on a fetcher: {result}")
                        continue

                    # result is a generator that outputs a pair [{'gpus': i/N}, {metric_key1: metric_state1, ...}}]
                    for gpu_info, local_metric_state in result.items():
                        if isinstance(local_metric_state, dict):
                            all_local_states.append(local_metric_state)
                        else:
                            logger.warning(
                                f"Unexpected result from fetcher. {gpu_info=}, {local_metric_state=}"
                            )
                return all_local_states

            all_local_states = extract_values_from_valuemesh(results)

            if not all_local_states:
                logger.warning(f"No states to reduce for global_step {global_step}")
                return

            # Reduce metrics from states
            reduced_metrics = reduce_metrics_states(all_local_states)

            # Split into scalar metrics and sample metrics
            scalar_metrics = [
                m for m in reduced_metrics if m.reduction != Reduce.SAMPLE
            ]
            sample_metrics = [
                m for m in reduced_metrics if m.reduction == Reduce.SAMPLE
            ]

            # Log to global backends
            for backend_name, backend in self.global_logger_backends.items():
                if scalar_metrics:
                    await backend.log_batch(scalar_metrics, global_step)
                if sample_metrics:
                    await backend.log_samples(sample_metrics, global_step)

    @endpoint
    async def has_fetcher(self, proc_id: str) -> bool:
        """Check if a fetcher is registered with the given proc_id."""
        return proc_id in self.fetchers

    @endpoint
    async def get_fetcher_count(self) -> int:
        return len(self.fetchers)

    @endpoint
    async def shutdown(self) -> None:
        if self.fetchers:
            try:
                tasks = [fetcher.shutdown.call() for fetcher in self.fetchers.values()]
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=2.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Metric logging fetcher shutdown timed out likely due to the child process being terminated before the parent."
                )

        # Finish global logger_backends
        for logger_backend_name, logger_backend in self.global_logger_backends.items():
            await logger_backend.finish()
