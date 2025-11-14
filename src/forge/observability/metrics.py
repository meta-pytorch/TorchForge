# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import heapq
import itertools
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import pytz

from forge.observability.utils import get_proc_name_with_rank

from forge.util.logging import get_logger, log_once
from monarch.actor import current_rank

logger = get_logger("INFO")


class BackendRole(Enum):
    """Backend role constants for metric logging actors.

    Defines whether an actor operates as a local (per-rank) or global (controller) role
    in the distributed metrics collection system.
    """

    LOCAL = "local"
    GLOBAL = "global"


class LoggingMode(Enum):
    """Metric logging behavior for distributed training scenarios.

    Each mode serves different observability needs:

    GLOBAL_REDUCE = "global_reduce"
        Best for: Metrics that are best visualized as a single value per step.
        Behavior: All ranks accumulate → controller reduces → single log entry
        Example use: 8 ranks training, want 1 loss value per training step averaged across all
        Where: GlobalLoggingActor logs reduced values to backends on flush.

    PER_RANK_REDUCE = "per_rank_reduce"
        Best for: Per-rank performance metrics, debugging individual rank behavior
        Behavior: Each rank accumulates + logs its own reduced values
        Example use: Monitor GPU utilization per rank, get 8 separate log entries per step
        Where: MetricCollector on each rank log reduced values to backends on flush.

    PER_RANK_NO_REDUCE = "per_rank_no_reduce"
        Best for: Real-time streaming, time-series debugging
        Behavior: Raw values logged immediately on record_metric() calls. Ignores reduce type.
        Example use: See what every rank is doing in real time.
        Where: MetricCollector on each rank log raw values to backends on push.
    """

    GLOBAL_REDUCE = "global_reduce"
    PER_RANK_REDUCE = "per_rank_reduce"
    PER_RANK_NO_REDUCE = "per_rank_no_reduce"


class Reduce(Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    STD = "std"
    SAMPLE = "sample"

    @property
    def accumulator_class(self):
        mapping = {
            Reduce.MEAN: MeanAccumulator,
            Reduce.SUM: SumAccumulator,
            Reduce.MAX: MaxAccumulator,
            Reduce.MIN: MinAccumulator,
            Reduce.STD: StdAccumulator,
            Reduce.SAMPLE: SampleAccumulator,
        }
        return mapping[self]


@dataclass
class Metric:
    """Container for metric data including key, value, reduction type, and timestamp.

    Timestamp is automatically set to current UTC time if not provided.
    """

    key: str
    value: Any
    reduction: Reduce
    timestamp: float | None = None

    def __post_init__(self):
        if self.timestamp is None:
            # Always record in UTC timezone
            self.timestamp = datetime.now(pytz.UTC).timestamp()


def record_metric(key: str, value: Any, reduction: Reduce = Reduce.MEAN) -> None:
    """Thin wrapper to send metrics to per-rank local MetricCollectors.

    Relies on a per-rank MetricCollector singleton for ease of use, i.e.
    call `record_metric` anywhere in the code without moving the
    collector from function to function.

    Can be disabled globally by setting the environment variable `FORGE_DISABLE_METRICS=true`.

    Collected metrics are flushed to backends on flush(), generally:
    GlobalLoggingActor.method() -> per-procmesh LocalFetcherActor.method() -> per-rank MetricCollector.method() -> logger
    """
    # Skip metrics collection
    if os.getenv("FORGE_DISABLE_METRICS", "false").lower() == "true":
        return

    # timestamp is added automatically by the Metric class
    metric = Metric(key=key, value=value, reduction=reduction)
    collector = MetricCollector()
    collector.push(metric)


def reduce_metrics_states(states: list[dict[str, dict[str, Any]]]) -> list[Metric]:
    """Reduce metric accumulators states to a list of metrics.

    Can be used when reducing metrics across ranks or services, as merging
    states is more precise than merging locally reduced metrics.

    Args:
        states (list[dict[str, dict[str, Any]]]): List of state of one or more metrics,
            normally retrieved using `forge.observability.metrics.MetricAccumulator.get_state()`.

    Returns:
        list[Metric]: List of reduced metrics

    Example:
        >>> states = [
        ...     {
        ...         "loss": {"count": 5, "sum": 14, "reduction_type": "mean"},
        ...         "reward/sample": {
        ...             "reduction_type": "sample",
        ...             "samples": [{"episode_id": 1, "reward": 0.5}],
        ...         },
        ...     },
        ...     {"loss": {"count": 10, "sum": 16, "reduction_type": Reduce.MEAN}},
        ... ]
        >>> reduce_metrics_states(states)
        [
            Metric(key='loss', value=2.0, reduction=Reduce.MEAN),
            Metric(
                key='reward/sample',
                value=[{'episode_id': 1, 'reward': 0.5}],
                reduction=Reduce.SAMPLE,
            )
        ]

    Raises:
        ValueError: on mismatched reduction types for the same metric key.
    """
    if not states:
        return []

    # Collect unique keys across all
    all_keys = set(k for state in states for k in state)

    reduced_metrics = []
    for key in all_keys:
        metric_states = [state.get(key) for state in states if key in state]
        if not metric_states:
            continue

        first_reduction_type = metric_states[0]["reduction_type"]  # pyre-ignore

        # Check consistency
        for state in metric_states:
            if state is None:
                continue
            if state["reduction_type"] != first_reduction_type:
                raise ValueError(
                    f"Mismatched reduction types for key '{key}': {first_reduction_type} vs {state['reduction_type']}"
                )

        metric_accumulator = Reduce(first_reduction_type).accumulator_class
        reduced_value = metric_accumulator.get_reduced_value_from_states(metric_states)

        # Create Metric object with reduced value
        metric = Metric(
            key=key,
            value=reduced_value,
            reduction=Reduce(first_reduction_type),
        )
        reduced_metrics.append(metric)

    return reduced_metrics


def record_episode_sample(table_name: str, episode):
    """
    Record a structured sample-level log for a single episode.
    Args:
        table_name (str): logging prefix (e.g. "rollout/sample").
        episode (Episode): episode object with filled attributes.
    """
    sample = episode.to_dict(exclude=["ref_logprobs", "completion"])
    record_metric(table_name, sample, Reduce.SAMPLE)


################
# Accumulators #
################


class MetricAccumulator(ABC):
    """Every metric maps to a MetricAccumulator, which accumulates values and optionally reduces them."""

    def __init__(self, reduction: Reduce) -> None:
        self.reduction_type = reduction
        self.is_reset = True

    @abstractmethod
    def append(self, value: Any) -> None:
        """Updates accumulator with new value (e.g., adds to sum and count for MEAN)."""
        pass

    @abstractmethod
    def get_value(self) -> Any:
        """Returns locally reduced value (e.g., sum/count for MEAN)."""
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Returns serializable state for cross-rank merge (e.g., {'sum': 10.0, 'count': 5})."""
        pass

    @classmethod
    @abstractmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> Any:
        """Merges states from multiple ranks into single reduced value (e.g., total_sum/total_count for MEAN)."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clears for next accumulation cycle (e.g., sum=0, count=0 for MEAN)."""
        pass


class MeanAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce) -> None:
        super().__init__(reduction)
        self.sum = 0.0
        self.count = 0
        self.is_reset = True

    def append(self, value: Any) -> None:
        v = float(value.item() if hasattr(value, "item") else value)
        self.is_reset = False
        self.sum += v
        self.count += 1

    def get_value(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def get_state(self) -> dict[str, Any]:
        return {
            "reduction_type": self.reduction_type.value,
            "sum": self.sum,
            "count": self.count,
        }

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        total_sum = sum(s["sum"] for s in states)
        total_count = sum(s["count"] for s in states)
        return total_sum / total_count if total_count > 0 else 0.0

    def reset(self) -> None:
        self.is_reset = True
        self.sum = 0.0
        self.count = 0


class SumAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce) -> None:
        super().__init__(reduction)
        self.total = 0.0
        self.is_reset = True

    def append(self, value: Any) -> None:
        v = float(value.item() if hasattr(value, "item") else value)
        self.is_reset = False
        self.total += v

    def get_value(self) -> float:
        return self.total

    def get_state(self) -> dict[str, Any]:
        return {"reduction_type": self.reduction_type.value, "total": self.total}

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        return sum(s["total"] for s in states)

    def reset(self) -> None:
        self.is_reset = True
        self.total = 0.0


class MaxAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce) -> None:
        super().__init__(reduction)
        self.max_val = float("-inf")
        self.is_reset = True

    def append(self, value: Any) -> None:
        v = float(value.item() if hasattr(value, "item") else value)
        self.is_reset = False
        self.max_val = max(self.max_val, v)

    def get_value(self) -> float:
        return self.max_val

    def get_state(self) -> dict[str, Any]:
        return {
            "reduction_type": self.reduction_type.value,
            "max_val": self.max_val,
        }

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        return max(s["max_val"] for s in states)

    def reset(self) -> None:
        self.is_reset = True
        self.max_val = float("-inf")


class MinAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce) -> None:
        super().__init__(reduction)
        self.min_val = float("inf")
        self.is_reset = True

    def append(self, value: Any) -> None:
        v = float(value.item() if hasattr(value, "item") else value)
        self.is_reset = False
        self.min_val = min(self.min_val, v)

    def get_value(self) -> float:
        return self.min_val

    def get_state(self) -> dict[str, Any]:
        return {
            "reduction_type": self.reduction_type.value,
            "min_val": self.min_val,
        }

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        return min(s["min_val"] for s in states)

    def reset(self) -> None:
        self.is_reset = True
        self.min_val = float("inf")


class StdAccumulator(MetricAccumulator):
    def __init__(self, reduction: Reduce) -> None:
        super().__init__(reduction)
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0
        self.is_reset = True

    def append(self, value: Any) -> None:
        v = float(value.item() if hasattr(value, "item") else value)
        self.is_reset = False
        self.sum += v
        self.sum_sq += v * v
        self.count += 1

    def get_value(self) -> float:
        if self.count == 0:
            return 0.0
        if self.count == 1:
            return 0.0
        mean = self.sum / self.count
        variance = (self.sum_sq / self.count) - (mean * mean)
        return max(0.0, variance) ** 0.5

    def get_state(self) -> dict[str, Any]:
        return {
            "reduction_type": self.reduction_type.value,
            "sum": self.sum,
            "sum_sq": self.sum_sq,
            "count": self.count,
        }

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        total_sum = sum(s["sum"] for s in states)
        total_sum_sq = sum(s["sum_sq"] for s in states)
        total_count = sum(s["count"] for s in states)
        if total_count == 0:
            return 0.0
        if total_count == 1:
            return 0.0
        mean = total_sum / total_count
        variance = (total_sum_sq / total_count) - (mean * mean)
        return max(0.0, variance) ** 0.5

    def reset(self) -> None:
        self.is_reset = True
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0


class SampleAccumulator(MetricAccumulator):
    """Accumulator for sample-level metrics with top-k and bottom-k filtering.

    Keeps the top-k and bottom-k samples by a given key (e.g., reward).
    Useful for logging only the best and worst samples from a batch.
    """

    def __init__(
        self, reduction: Reduce, top_k: int = 1, bottom_k: int = 1, key: str = "reward"
    ):
        super().__init__(reduction)
        self.samples: List[Dict[str, Any]] = []
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.key = key
        self._top_heap = []  # min-heap for top-k
        self._bottom_heap = []  # max-heap for bottom-k (store -value)
        self._counter = itertools.count()  # tie-breaker id generator
        self.is_reset = True

    def append(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value)}")

        self.is_reset = False
        val = value.get(self.key, 0.0)
        idx = next(self._counter)  # unique tiebreaker

        # If top_k or bottom_k <= 0, it means "disable" that side of filtering (i.e., keep none).
        # maintain top-k
        if self.top_k > 0:
            if len(self._top_heap) < self.top_k:
                heapq.heappush(self._top_heap, (val, idx, value))
            else:
                heapq.heappushpop(self._top_heap, (val, idx, value))

        # maintain bottom-k
        if self.bottom_k > 0:
            if len(self._bottom_heap) < self.bottom_k:
                heapq.heappush(self._bottom_heap, (-val, idx, value))
            else:
                heapq.heappushpop(self._bottom_heap, (-val, idx, value))

    def get_value(self) -> list[dict]:
        """Return top-k and bottom-k filtered samples."""
        tops = [s for _, _, s in self._top_heap]
        bottoms = [s for _, _, s in self._bottom_heap]
        return bottoms + tops

    def get_state(self) -> Dict[str, Any]:
        """Serialize accumulator state for cross-rank reduction."""
        return {
            "reduction_type": self.reduction_type.value,
            "samples": self.get_value(),
        }

    @classmethod
    def get_reduced_value_from_states(cls, states: List[Dict[str, Any]]) -> list[dict]:
        """Merge sample states across ranks."""
        merged = []
        for s in states:
            merged.extend(s.get("samples", []))
        return merged

    def reset(self) -> None:
        """Clear local samples and reset filter state."""
        self.is_reset = True
        self.samples.clear()
        self._top_heap = []
        self._bottom_heap = []
        self._counter = itertools.count()


#############
# Collector #
#############


class MetricCollector:
    """Per-rank singleton for accumulating, retrieving and flushing metrics to backends.

    Supports multiple logging backends, each with different logging modes.
    For options, check `forge.observability.metrics.LoggerBackend` and `forge.observability.metrics.LoggingMode`.

    Behavior:
    - Ensures one instance per rank;
    - Using `record_metric()` delegates here;
    - Init via GlobalLoggingActor -> LocalFetcherActor -> per-rank MetricCollector;
    - GlobalLoggingActor flushes trigger reductions and log for any locally setup backend. Can optionally also
    return non-reduced states for global aggregation.
    - Resets accumulators post-flush to avoid leaks across steps;
    """

    _instances: dict[int, "MetricCollector"] = {}
    _singleton_rank: int

    def __new__(cls):
        """Singleton per-rank, ensures one instance per rank."""
        rank = current_rank().rank

        if rank not in cls._instances:
            inst = super().__new__(cls)
            cls._instances[rank] = inst
            inst._singleton_rank = rank
        else:
            inst = cls._instances[rank]
            # Defensive check for bugs in singleton implementation - should never fail in normal operation
            if inst._singleton_rank != rank:
                raise ValueError(
                    f"Singleton expected rank {inst._singleton_rank}, but saw {rank}"
                )
        return inst

    def __init__(self) -> None:
        if hasattr(self, "_is_initialized"):
            return

        self.accumulators: dict[str, MetricAccumulator] = {}
        self.rank = current_rank().rank
        self.per_rank_reduce_backends: list[LoggerBackend] = []
        self.per_rank_no_reduce_backends: list[LoggerBackend] = []
        self.global_step: int = 0  # Set on `init_backends` and updated on `flush`
        self._is_initialized = False
        self.proc_name_with_rank: str | None = None

    async def init_backends(
        self,
        metadata_per_controller_backend: dict[str, dict[str, Any]] | None,
        config: dict[str, Any],
        global_step: int = 0,
        process_name: str | None = None,
    ) -> None:
        """Initialize per-rank logger backends and MetricCollector state.

        A logger backend is represented by a backend class (e.g. WandBBackend, ConsoleBackend).
        Backends are categorized by their logging_mode. For details, see `forge.observability.metrics.LoggingMode`.

        Args:
            metadata_per_controller_backend (Optional[Dict[str, Dict[str, Any]]]): Metadata from controller
                for backends that require shared state across processes, e.g.,
                {"wandb": {"shared_run_id": "abc123"}}.
            config (Dict[str, Any]): Backend configurations where each key is a backend name
                and value contains logging_mode and backend-specific settings.
                e.g., {"wandb": {"logging_mode": "per_rank_no_reduce", "project": "my_proj"}}
            global_step (int, default 0): Initial step for logging. Can be used when
                resuming from a checkpoint.
            process_name (str | None): The meaningful process name for logging.
        """
        if self._is_initialized:
            logger.debug(
                f"{self.proc_name_with_rank}: MetricCollector already initialized"
            )
            return

        self.global_step = global_step
        self.proc_name_with_rank = get_proc_name_with_rank(process_name)

        self.per_rank_reduce_backends: list[LoggerBackend] = []
        self.per_rank_no_reduce_backends: list[LoggerBackend] = []

        # Initialize backends based on logging mode
        for backend_name, backend_config in config.items():
            mode = backend_config["logging_mode"]

            # sanity check
            if not isinstance(mode, LoggingMode):
                raise TypeError(
                    f"Expected LoggingMode enum for {backend_name}.logging_mode, got {type(mode)}: {mode}."
                )

            # We should never hit this. Backend will be instantiated in GlobalLoggingActor.
            if mode == LoggingMode.GLOBAL_REDUCE:
                logger.debug("Skipping local instantiation for GLOBAL_REDUCE.")
                continue

            # get metadata from controller backend, if any
            controller_metadata = {}
            if metadata_per_controller_backend:
                controller_metadata = metadata_per_controller_backend.get(
                    backend_name, {}
                )

            # instantiate local backend
            backend: LoggerBackend = get_logger_backend_class(backend_name)(
                **backend_config
            )
            await backend.init(
                role=BackendRole.LOCAL,
                controller_logger_metadata=controller_metadata,
                process_name=self.proc_name_with_rank,
            )

            # Categorize by logging mode
            if mode == LoggingMode.PER_RANK_NO_REDUCE:
                self.per_rank_no_reduce_backends.append(backend)
            else:
                self.per_rank_reduce_backends.append(backend)

        self._is_initialized = True

    def push(self, metric: Metric) -> None:
        """Process a metric according to configured logging modes.

        Behavior depends on backend modes:
        - PER_RANK_NO_REDUCE: Stream metric immediately to backends
        - PER_RANK_REDUCE/GLOBAL_REDUCE: Accumulate for per step batch logging

        Args:
            metric (Metric): Metric dataclass

        Example:
            collector = MetricCollector()
            metric = Metric("loss", 0.5, Reduce.MEAN)
            collector.push(metric)  # Streams immediately if no_reduce, else accumulates
        """
        # sanity check
        if not self._is_initialized:
            log_once(
                logger,
                level=logging.WARNING,
                msg=(
                    f"Skipping metric collection for {get_proc_name_with_rank()}."
                    " Metric logging backends (e.g. wandb) were not initialized."
                    " This happens when you try to use `record_metric` before calling `init_backends`."
                    " To disable this warning, please call in your main file:\n"
                    "`mlogger = await get_or_create_metric_logger(process_name='Controller')`\n"
                    "`await mlogger.init_backends.call_one(logging_config)`\n"
                    "or set env variable `FORGE_DISABLE_METRICS=True`"
                ),
            )
            return

        # Validate metric object
        if not isinstance(metric, Metric):
            raise TypeError(
                f"Expected {Metric} object, got {metric} of type {type(metric)}"
            )

        # For PER_RANK_NO_REDUCE backends: stream without reduce
        for backend in self.per_rank_no_reduce_backends:
            if metric.reduction == Reduce.SAMPLE:
                asyncio.create_task(backend.log_samples([metric], self.global_step))
            else:
                backend.log_stream(metric=metric, global_step=self.global_step)

        # Always accumulate for reduction and state return
        key = metric.key
        if key not in self.accumulators:
            self.accumulators[key] = metric.reduction.accumulator_class(
                metric.reduction
            )
        self.accumulators[key].append(metric.value)

    async def flush(
        self, global_step: int, return_state: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Log to local logger backends (if any), reset accumulators and return metric states dict if return_state=True.

        Args:
            global_step (int): step used by backends to align metrics on the same x-axis
            return_state (bool): Used by GlobalLoggingActor for reduction across all ranks.
                If False, returns empty dict, else returns the state of all metrics collected.
        Returns:
            dict[str, dict[str, Any]]: Dict of {metric_key: metric_state},
                e.g., {"loss": {"reduction_type": "mean", "sum": 1.2, "count": 3}}.
        """
        if not self._is_initialized:
            log_once(
                logger,
                level=logging.WARNING,
                msg=f"Cannot flush collected metrics for {get_proc_name_with_rank()}. "
                " MetricCollector.flush() called before init_backends()."
                "\nPlease call in your main file:\n"
                "`mlogger = await get_or_create_metric_logger(process_name='Controller')`\n"
                "`await mlogger.init_backends.call_one(logging_config)`\n"
                "before calling `flush`",
            )
            return {}

        if not self.accumulators:
            logger.debug(
                f"Collector {self.proc_name_with_rank}: No metrics to flush for global_step {global_step}"
            )
            return {}

        # Snapshot states and reset immediately
        states = {}
        for key, acc in self.accumulators.items():
            # Skip state if nothing was accumulated
            if acc.is_reset:
                continue
            states[key] = acc.get_state()
            acc.reset()

        # Reduce and log to PER_RANK_REDUCE backends only (NO_REDUCE backends already logged in push)
        if self.per_rank_reduce_backends:
            metrics_for_backends = reduce_metrics_states([states])

            # Split into scalar metrics and sample metrics
            scalar_metrics = [
                m for m in metrics_for_backends if m.reduction != Reduce.SAMPLE
            ]
            sample_metrics = [
                m for m in metrics_for_backends if m.reduction == Reduce.SAMPLE
            ]

            for backend in self.per_rank_reduce_backends:
                if scalar_metrics:
                    await backend.log_batch(scalar_metrics, global_step)
                if sample_metrics:
                    await backend.log_samples(sample_metrics, global_step)

        # Update step counter for streaming backends
        # Note: This is incremented AFTER flush completes, so metrics recorded between
        # flush(N) and flush(N+1) will stream with global_step=N+1.
        self.global_step = global_step + 1

        return states if return_state else {}

    async def shutdown(self):
        """Shutdown logger_backends if initialized."""

        if not self._is_initialized:
            logger.debug(
                f"Collector for {self.proc_name_with_rank} not initialized. Skipping shutdown"
            )
            return

        for backend in self.per_rank_reduce_backends + self.per_rank_no_reduce_backends:
            await backend.finish()


###########
# Backends #
###########


class LoggerBackend(ABC):
    """Abstract logger_backend for metric logging, e.g. wandb, jsonl, etc.

    Args:
        logging_mode: Logging behavior mode.
        per_rank_share_run: Whether ranks share run. Default False.
        **kwargs: Backend-specific arguments (e.g., project, name, tags for WandB).
    """

    def __init__(
        self, *, logging_mode: LoggingMode, per_rank_share_run: bool = False, **kwargs
    ) -> None:

        self.logging_mode = logging_mode
        self.per_rank_share_run = per_rank_share_run
        self.backend_kwargs = kwargs

    @abstractmethod
    async def init(
        self,
        role: BackendRole,
        controller_logger_metadata: dict[str, Any] | None = None,
        process_name: str | None = None,
    ) -> None:
        """
        Initializes backend, e.g. wandb.run.init().

        Args:
            role (BackendRole): BackendRole.GLOBAL (controller) or BackendRole.LOCAL (per-rank).
                Can be used to behave differently for controller vs rank roles.
            controller_logger_metadata (dict[str, Any] | None): From global backend for
                backend that required shared info, e.g. {"shared_run_id": "abc123"}.
            process_name (str | None): Process name for logging.

        Raises: ValueError if missing metadata for shared local init.
        """
        pass

    @abstractmethod
    async def log_batch(
        self, metrics: list[Metric], global_step: int, *args, **kwargs
    ) -> None:
        """Log batch of accumulated metrics to backend

        Args:
            metrics: List of Metric objects to log.
            global_step: Step number for x-axis alignment across metrics."""
        pass

    def log_stream(self, metric: Metric, global_step: int, *args, **kwargs) -> None:
        """Stream single metric to backend immediately.

        NOTE: This method is called synchronously.
        If your backend requires async I/O operations:
        - Use asyncio.create_task() for fire-and-forget logging
        - Consider internal buffering to avoid blocking the caller

        Example for async backend:
            def log_stream(self, metric, global_step):
                asyncio.create_task(self._async_log(metric, global_step))
        """
        pass

    @abstractmethod
    async def finish(self) -> None:
        pass

    def get_metadata_for_secondary_ranks(self) -> dict[str, Any] | None:
        """Return sharable state after controller init (e.g., for shared modes). Called only on controller backends."""
        return None


class ConsoleBackend(LoggerBackend):
    """Simple console logging of metrics."""

    def __init__(
        self, *, logging_mode: LoggingMode, per_rank_share_run: bool = False, **kwargs
    ) -> None:
        super().__init__(
            logging_mode=logging_mode, per_rank_share_run=per_rank_share_run, **kwargs
        )
        self.process_name = None

    async def init(
        self,
        role: BackendRole,
        controller_logger_metadata: dict[str, Any] | None = None,
        process_name: str | None = None,
    ) -> None:
        self.process_name = process_name

    async def log_batch(
        self, metrics: list[Metric], global_step: int, *args, **kwargs
    ) -> None:
        metrics_str = "\n".join(
            f"  {metric.key}: {metric.value}"
            for metric in sorted(metrics, key=lambda m: m.key)
        )
        logger.info(
            f"=== [{self.process_name}] - METRICS STEP {global_step} ===\n{metrics_str}\n==============================\n"
        )

    def log_stream(self, metric: Metric, global_step: int, *args, **kwargs) -> None:
        logger.info(f"{metric.key}: {metric.value}")

    async def finish(self) -> None:
        pass

    async def log_samples(self, samples: List[Metric], step: int) -> None:
        """Pretty-print sample-level logs to console."""

        logger.info(f"==========  SAMPLE LOGS STEP {step} ==========")
        for sample in samples:
            table_name, table_rows = sample.key, sample.value
            logger.info(f"[{table_name}] ({len(table_rows)} samples)")
            logger.info(json.dumps(table_rows, indent=2, ensure_ascii=False))
        logger.info("==============================================\n")


class WandbBackend(LoggerBackend):
    """
    Weights & Biases logging backend.

    For logging mode details, see `forge.observability.metrics.LoggingMode` documentation.

    More details on wandb distributed logging: https://docs.wandb.ai/guides/track/log/distributed-training/

    Configuration:
        logging_mode (LoggingMode): Determines logging behavior.
        per_rank_share_run (bool, default False): For per-rank modes, whether to share run ID across ranks.
            If true, a single wandb run is created and all ranks log to it. Particularly useful for
            logging with no_reduce to capture time-based streams. Not recommended if reducing values.
        **kwargs: Any argument accepted by wandb.init() (e.g., project, group, name, tags, notes, etc.)

    Example:
        WandbBackend(
            logging_mode=LoggingMode.PER_RANK_REDUCE,
            per_rank_share_run=False,
            project="my_project",
            group="exp_group",
            name="my_experiment",
            tags=["rl", "v2"],
            notes="Testing new reward"
        )
    """

    def __init__(
        self, *, logging_mode: LoggingMode, per_rank_share_run: bool = False, **kwargs
    ) -> None:
        super().__init__(
            logging_mode=logging_mode, per_rank_share_run=per_rank_share_run, **kwargs
        )
        self.run = None
        self.process_name = None
        self._tables: dict[str, "wandb.Table"] = {}

    async def init(
        self,
        role: BackendRole,
        controller_logger_metadata: dict[str, Any] | None = None,
        process_name: str | None = None,
    ) -> None:
        if controller_logger_metadata is None:
            controller_logger_metadata = {}

        # Pop name, if any, to concat to process_name.
        run_name = self.backend_kwargs.pop("name", None)
        self.process_name = process_name

        # Format run name based on mode and role
        if self.logging_mode == LoggingMode.GLOBAL_REDUCE:
            if role != BackendRole.GLOBAL:
                logger.warning(f"Skipped init for GLOBAL_REDUCE mode and {role} role.")
                return
            # use name as-is, no need to append controller process_name
            await self._init_global(run_name)

        elif role == BackendRole.GLOBAL and self.per_rank_share_run:
            # use name as-is, no need to append controller process_name
            await self._init_shared_global(run_name)

        elif role == BackendRole.LOCAL:
            # Per-rank: append process_name
            run_name = f"{run_name}_{process_name}" if run_name else process_name

            if self.per_rank_share_run:
                shared_id = controller_logger_metadata.get("shared_run_id")
                if shared_id is None:
                    raise ValueError(
                        f"Shared ID required but not provided for {process_name} backend init"
                    )
                await self._init_shared_local(run_name, shared_id, process_name)
            else:
                await self._init_per_rank(run_name)

    async def _init_global(self, run_name: str | None):
        import wandb

        self.run = wandb.init(name=run_name, **self.backend_kwargs)

    async def _init_per_rank(self, run_name: str):
        import wandb

        self.run = wandb.init(name=run_name, **self.backend_kwargs)

    async def _init_shared_global(self, run_name: str | None):
        import wandb

        settings = wandb.Settings(
            mode="shared", x_primary=True, x_label="controller_primary"
        )
        self.run = wandb.init(name=run_name, settings=settings, **self.backend_kwargs)

    async def _init_shared_local(
        self, run_name: str, shared_id: str, process_name: str
    ):
        import wandb

        # Clear any stale service tokens that might be pointing to dead processes
        # In multiprocessing environments, WandB service tokens can become stale and point
        # to dead service processes. This causes wandb.init() to hang indefinitely trying
        # to connect to non-existent services. Clearing forces fresh service connection.
        from wandb.sdk.lib.service import service_token

        service_token.clear_service_in_env()

        settings = wandb.Settings(mode="shared", x_primary=False, x_label=process_name)
        self.run = wandb.init(
            name=run_name, id=shared_id, settings=settings, **self.backend_kwargs
        )

    async def log_batch(
        self, metrics: list[Metric], global_step: int, *args, **kwargs
    ) -> None:
        if not self.run:
            logger.debug(
                f"WandbBackend: No run started, skipping log for {self.process_name}"
            )
            return

        # Convert metrics to WandB log format
        log_data = {}
        for metric in metrics:
            log_data[metric.key] = metric.value

        self.run.log(log_data, step=global_step)
        logger.info(
            f"WandbBackend: Logged {len(metrics)} metrics at step {global_step}"
        )

    def log_stream(self, metric: Metric, global_step: int, *args, **kwargs) -> None:
        """Stream single metric to WandB with both step and timestamp."""
        if not self.run:
            return

        # Log with custom timestamp for precision
        # Users can choose x-axis as timestamp in WandB UI and display as datetime
        log_data = {
            metric.key: metric.value,
            "timestamp": metric.timestamp,
        }

        # note: here we dont use step since wandb keeps only the latest value for each step
        self.run.log(log_data)

    async def log_samples(self, samples: List[Metric], step: int) -> None:
        """Log sample-level data incrementally to persistent WandB Tables."""
        import wandb

        if not self.run:
            return

        for sample in samples:
            table_name, table_rows = sample.key, sample.value
            if not table_rows:
                continue

            # Convert to list if single sample. This happens when logging stream
            if isinstance(table_rows, dict):
                table_rows = [table_rows]

            # If table doesn't exist yet, create it in INCREMENTAL mode
            if table_name not in self._tables:
                # Collect all unique columns from all rows
                columns = set()
                for row in table_rows:
                    columns.update(row.keys())
                columns = sorted(columns)  # Sort for consistent column ordering
                table = wandb.Table(columns=columns, log_mode="INCREMENTAL")
                self._tables[table_name] = table
                logger.debug(
                    f"WandbBackend: Created new incremental table: {table_name} with columns: {columns}"
                )
            else:
                table = self._tables[table_name]

            # Add rows (fill missing columns with None)
            for s in table_rows:
                values = [s.get(c) for c in table.columns]
                table.add_data(*values)

            # Log the same table object (INCREMENTAL update)
            # table_name has to end with _table to be recognized by wandb
            if not table_name.endswith("_table"):
                table_name += "_table"
            self.run.log({f"{table_name}": table})

    def get_metadata_for_secondary_ranks(self) -> dict[str, Any]:
        if self.run and self.per_rank_share_run:
            return {"shared_run_id": self.run.id}
        return {}

    async def finish(self) -> None:
        import wandb

        if self.run:
            """
            Convert each incremental table to immutable before finishing
            as recommended by wandb:
            https://docs.wandb.ai/models/tables/log_tables#incremental-mode
            """
            for table_name, incr_table in self._tables.items():
                final_table = wandb.Table(
                    columns=incr_table.columns,
                    data=incr_table.data,
                    log_mode="IMMUTABLE",
                )
                self.run.log({table_name: final_table})
                logger.debug(f"WandbBackend: Finalized table {table_name}")

            self.run.finish()
            logger.info(f"WandbBackend {self.process_name}: Finished run")


def get_logger_backend_class(cls_name: str) -> type[LoggerBackend]:
    """Simple mapping between logger_backend type and its class

    Factory for backend classes from config; returns uninitialized class for role-based init.
    """
    if cls_name == "console":
        return ConsoleBackend
    elif cls_name == "wandb":
        return WandbBackend
    else:
        raise ValueError(f"Unknown logger backend type: {cls_name}")
