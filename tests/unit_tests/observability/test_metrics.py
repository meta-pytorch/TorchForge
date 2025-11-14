# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for core metrics functionality."""

import time
from unittest.mock import MagicMock, patch

import pytest

from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import (
    BackendRole,
    ConsoleBackend,
    get_logger_backend_class,
    LoggingMode,
    MaxAccumulator,
    MeanAccumulator,
    Metric,
    MetricCollector,
    MinAccumulator,
    record_metric,
    Reduce,
    reduce_metrics_states,
    StdAccumulator,
    SumAccumulator,
    WandbBackend,
)


class TestMetricCreation:
    """Test Metric object creation and record_metric function - Diff 2 features."""

    def test_metric_creation_automatic_timestamp(self, mock_rank):
        """Test Metric object creation with automatic timestamp."""
        before_time = time.time()
        metric = Metric("test_key", 42.0, Reduce.MEAN)
        after_time = time.time()

        assert metric.key == "test_key"
        assert metric.value == 42.0
        assert metric.reduction == Reduce.MEAN
        assert metric.timestamp is not None
        assert before_time <= metric.timestamp <= after_time

    def test_metric_creation_custom_timestamp(self, mock_rank):
        """Test Metric object creation with custom timestamp."""
        custom_time = 1234567890.0
        metric = Metric("test_key2", 24.0, Reduce.SUM, timestamp=custom_time)
        assert metric.timestamp == custom_time

    def test_record_metric(self, mock_rank):
        """Test record_metric creates correct Metric and calls collector."""
        # Mock the MetricCollector constructor to return a mock instance
        mock_collector = MagicMock()

        with patch(
            "forge.observability.metrics.MetricCollector", return_value=mock_collector
        ):
            record_metric("loss", 1.5, Reduce.MEAN)

            # Verify push was called on the mock collector
            mock_collector.push.assert_called_once()

            # Verify the metric passed to push
            pushed_metric = mock_collector.push.call_args[0][0]
            assert pushed_metric.key == "loss"
            assert pushed_metric.value == 1.5
            assert pushed_metric.reduction == Reduce.MEAN

    def test_new_enums_and_constants(self):
        """Test BackendRole constants and usage."""
        # Test BackendRole enum values
        assert BackendRole.LOCAL.value == "local"
        assert BackendRole.GLOBAL.value == "global"

        # Test that BackendRole is a proper Enum
        assert isinstance(BackendRole.LOCAL, BackendRole)
        assert isinstance(BackendRole.GLOBAL, BackendRole)

    @pytest.mark.asyncio
    async def test_backend_role_usage(self):
        """Test that BackendRole constants are actually used instead of string literals."""
        # Test ConsoleBackend
        console_backend = ConsoleBackend(logging_mode=LoggingMode.GLOBAL_REDUCE)
        await console_backend.init(role=BackendRole.LOCAL)

        # Test WandbBackend role validation without WandB initialization
        wandb_backend = WandbBackend(
            logging_mode=LoggingMode.GLOBAL_REDUCE, project="test"
        )

        # Mock all the WandB init methods to focus only on role validation
        with patch.object(wandb_backend, "_init_global"), patch.object(
            wandb_backend, "_init_shared_global"
        ), patch.object(wandb_backend, "_init_shared_local"), patch.object(
            wandb_backend, "_init_per_rank"
        ):

            # Should not raise error for valid roles (type system prevents invalid values)
            await wandb_backend.init(role=BackendRole.GLOBAL)
            await wandb_backend.init(role=BackendRole.LOCAL)


class TestReduceOperations:
    """Test reduce_metrics_states function returning List[Metric] - Diff 2 feature."""

    def test_empty_states(self):
        """Test reduce_metrics_states with empty input."""
        result = reduce_metrics_states([])
        assert result == []

    def test_single_state(self):
        """Test reduce_metrics_states with single state."""
        states = [
            {
                "loss": {"reduction_type": "mean", "sum": 10.0, "count": 2},
                "rollout/sample": {
                    "reduction_type": "sample",
                    "samples": [{"id": 1, "reward": 0.5}],
                },
            }
        ]
        metrics = reduce_metrics_states(states)
        assert len(metrics) == 2
        # Convert to dict for easier testing
        result_dict = {m.key: (m.value, m.reduction) for m in metrics}

        assert result_dict["loss"][0] == 5.0
        assert result_dict["loss"][1] == Reduce.MEAN

        assert result_dict["rollout/sample"][0] == [{"id": 1, "reward": 0.5}]
        assert result_dict["rollout/sample"][1] == Reduce.SAMPLE

    def test_multiple_states(self):
        """Test reduce_metrics_states with multiple states."""
        states = [
            {
                "loss": {"reduction_type": "mean", "sum": 10.0, "count": 2},
                "rollout/sample": {
                    "reduction_type": "sample",
                    "samples": [{"id": 1, "reward": 0.5}],
                },
            },
            {
                "loss": {"reduction_type": "mean", "sum": 20.0, "count": 3},
                "rollout/sample": {
                    "reduction_type": "sample",
                    "samples": [{"id": 2, "reward": 0.8}],
                },
            },
            {"accuracy": {"reduction_type": "sum", "total": 15.0}},
        ]
        metrics = reduce_metrics_states(states)

        assert len(metrics) == 3

        # Convert to dict for easier testing
        result_dict = {m.key: (m.value, m.reduction) for m in metrics}

        # Check scalar reductions
        assert result_dict["loss"][0] == 30.0 / 5.0  # 6.0
        assert result_dict["loss"][1] == Reduce.MEAN
        assert result_dict["accuracy"][0] == 15.0
        assert result_dict["accuracy"][1] == Reduce.SUM

        # Check sample concatenation
        assert result_dict["rollout/sample"][0] == [
            {"id": 1, "reward": 0.5},
            {"id": 2, "reward": 0.8},
        ]
        assert result_dict["rollout/sample"][1] == Reduce.SAMPLE

    def test_mismatched_reduction_types_raises_error(self):
        """Test reduce_metrics_states raises error for mismatched reduction types."""
        states = [
            {"loss": {"reduction_type": "mean", "sum": 10.0, "count": 2}},
            {"loss": {"reduction_type": "sum", "total": 20.0}},
        ]
        with pytest.raises(ValueError, match="Mismatched reduction types"):
            reduce_metrics_states(states)


class TestAccumulators:
    """Test all accumulator classes and their operations - Diff 2 extensions."""

    def test_sum_accumulator(self):
        """Test SumAccumulator operations."""
        acc = SumAccumulator(Reduce.SUM)

        acc.append(5.0)
        acc.append(3.0)
        assert acc.get_value() == 8.0

        state = acc.get_state()
        assert state["total"] == 8.0
        assert state["reduction_type"] == "sum"

        acc.reset()
        assert acc.get_value() == 0.0

    def test_max_accumulator(self):
        """Test MaxAccumulator operations."""
        acc = MaxAccumulator(Reduce.MAX)

        acc.append(5.0)
        acc.append(10.0)
        acc.append(3.0)
        assert acc.get_value() == 10.0

        state = acc.get_state()
        assert state["max_val"] == 10.0
        assert state["reduction_type"] == "max"

    def test_min_accumulator(self):
        """Test MinAccumulator operations."""
        acc = MinAccumulator(Reduce.MIN)

        acc.append(5.0)
        acc.append(10.0)
        acc.append(3.0)
        assert acc.get_value() == 3.0

        state = acc.get_state()
        assert state["min_val"] == 3.0
        assert state["reduction_type"] == "min"

    def test_std_accumulator(self):
        """Test StdAccumulator operations."""
        acc = StdAccumulator(Reduce.STD)

        # Test with zero/one values
        assert acc.get_value() == 0.0
        acc.append(5.0)
        assert acc.get_value() == 0.0  # std of single value is 0

        # Test with multiple values
        acc.append(7.0)  # values: 5, 7, mean=6, std=1
        assert abs(acc.get_value() - 1.0) < 0.001

        state = acc.get_state()
        assert state["sum"] == 12.0
        assert state["sum_sq"] == 74.0  # 5^2 + 7^2 = 25 + 49 = 74
        assert state["count"] == 2

    @pytest.mark.parametrize(
        "accumulator_class,states,expected",
        [
            (
                MeanAccumulator,
                [
                    {"reduction_type": "mean", "sum": 10.0, "count": 2},
                    {"reduction_type": "mean", "sum": 20.0, "count": 3},
                ],
                6.0,  # (10+20) / (2+3)
            ),
            (
                SumAccumulator,
                [
                    {"reduction_type": "sum", "total": 10.0},
                    {"reduction_type": "sum", "total": 15.0},
                ],
                25.0,
            ),
        ],
    )
    def test_accumulator_state_reduction(self, accumulator_class, states, expected):
        """Test cross-accumulator state reduction."""
        result = accumulator_class.get_reduced_value_from_states(states)
        assert result == expected

    def test_reduce_enum_accumulator_mapping(self):
        """Test that Reduce enum correctly maps to accumulator classes."""
        assert Reduce.MEAN.accumulator_class == MeanAccumulator
        assert Reduce.SUM.accumulator_class == SumAccumulator
        assert Reduce.MAX.accumulator_class == MaxAccumulator
        assert Reduce.MIN.accumulator_class == MinAccumulator
        assert Reduce.STD.accumulator_class == StdAccumulator


class TestCriticalFixes:
    """Test critical production fixes from Diff 1."""

    def test_uninitialized_push_logs_warning(self, mock_rank, caplog):
        """Test MetricCollector.push() logs warning when uninitialized."""
        collector = MetricCollector()
        metric = Metric("test", 1.0, Reduce.MEAN)

        # Should not raise error, just log warning and return
        collector.push(metric)
        assert any(
            "Metric logging backends" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_uninitialized_flush_logs_warning(self, mock_rank, caplog):
        """Test MetricCollector.flush() logs warning when uninitialized."""
        collector = MetricCollector()

        # Should not raise error, just log warning and return empty dict
        result = await collector.flush(global_step=1, return_state=True)
        assert result == {}
        assert any(
            "Cannot flush collected metrics" in record.message
            for record in caplog.records
        )

    @patch.dict("os.environ", {"FORGE_DISABLE_METRICS": "true"})
    @patch("forge.observability.metrics.MetricCollector")
    def test_record_metric_disabled(self, mock_collector_class):
        """Test record_metric is no-op when FORGE_DISABLE_METRICS=true."""
        record_metric("loss", 1.5, Reduce.MEAN)
        mock_collector_class.assert_not_called()

    @patch.dict("os.environ", {"FORGE_DISABLE_METRICS": "false"})
    @patch("forge.observability.metrics.MetricCollector")
    def test_record_metric_enabled_explicit(self, mock_collector_class, mock_rank):
        """Test record_metric works when FORGE_DISABLE_METRICS=false."""
        mock_collector = MagicMock()
        mock_collector_class.return_value = mock_collector

        record_metric("loss", 1.5, Reduce.MEAN)
        mock_collector_class.assert_called_once()
        mock_collector.push.assert_called_once()

    def test_wandb_backend_creation(self):
        """Test WandbBackend creation and basic setup without WandB dependency."""

        backend = WandbBackend(
            logging_mode=LoggingMode.GLOBAL_REDUCE,
            project="test_project",
            group="test_group",
        )

        # Test backend kwargs storage
        assert backend.backend_kwargs["project"] == "test_project"
        assert backend.backend_kwargs["group"] == "test_group"
        assert backend.logging_mode == LoggingMode.GLOBAL_REDUCE
        assert backend.per_rank_share_run is False  # default

        # Test metadata method
        metadata = backend.get_metadata_for_secondary_ranks()
        assert metadata == {}  # Should be empty when no run

    @pytest.mark.asyncio
    async def test_console_backend(self):
        """Test ConsoleBackend basic operations."""
        backend = ConsoleBackend(logging_mode=LoggingMode.GLOBAL_REDUCE)

        await backend.init(role=BackendRole.LOCAL)

        # Test log_batch - should not raise
        # Create a test metric
        test_metric = Metric("test", 1.0, Reduce.MEAN)
        await backend.log_batch([test_metric], global_step=1)

        await backend.finish()  # Should not raise


class TestBasicAccumulators:
    """Test basic accumulator functionality."""

    def test_mean_accumulator(self):
        """Test MeanAccumulator operations."""
        acc = MeanAccumulator(Reduce.MEAN)

        # Test initial state
        assert acc.get_value() == 0.0
        state = acc.get_state()
        assert state["sum"] == 0.0
        assert state["count"] == 0

        # Test append and get_value
        acc.append(10.0)
        acc.append(20.0)
        assert acc.get_value() == 15.0

        # Test state
        state = acc.get_state()
        assert state["sum"] == 30.0
        assert state["count"] == 2
        assert state["reduction_type"] == "mean"

        # Test reset
        acc.reset()
        assert acc.get_value() == 0.0
        assert acc.get_state()["sum"] == 0.0
        assert acc.get_state()["count"] == 0

    def test_reduce_enum_accumulator_mapping(self):
        """Test that Reduce enum correctly maps to accumulator classes."""
        assert Reduce.MEAN.accumulator_class == MeanAccumulator


class TestBackendFactory:
    """Test backend factory function."""

    def test_backend_factory(self):
        """Test get_logger_backend_class factory function."""
        assert get_logger_backend_class("console") == ConsoleBackend
        assert get_logger_backend_class("wandb") == WandbBackend

        with pytest.raises(ValueError, match="Unknown logger backend type"):
            get_logger_backend_class("invalid_backend")


class TestMetricCollector:
    """Test MetricCollector singleton behavior."""

    def test_singleton_per_rank(self, mock_rank):
        """Test MetricCollector singleton behavior per rank."""
        mock_rank.return_value.rank = 0
        collector1 = MetricCollector()
        collector2 = MetricCollector()
        assert collector1 is collector2

        # Different rank should get different instance
        mock_rank.return_value.rank = 1
        collector3 = MetricCollector()
        assert collector1 is not collector3


class TestMetricActorDisabling:
    """Test environment flag to disable metric actors."""

    async def _test_fetcher_registration(self, env_var_value, should_register_fetchers):
        """Check if FORGE_DISABLE_METRICS=[True, False, None] correctly disables fetcher registration.

        Args:
            env_var_value: Value to set for FORGE_DISABLE_METRICS (None means unset)
            should_register_fetchers: Whether fetchers should be registered (True) or not (False)
        """
        import os

        import forge.observability.metric_actors
        from forge.env import FORGE_DISABLE_METRICS
        from monarch.actor import this_host

        # set fresh env
        # Note: Environment variable setup is handled by clean_metrics_environment fixture
        forge.observability.metric_actors._global_logger = None

        if env_var_value is not None:
            os.environ[FORGE_DISABLE_METRICS.name] = env_var_value

        procs = this_host().spawn_procs(per_host={"cpus": 1})

        if hasattr(procs, "_local_fetcher"):
            delattr(procs, "_local_fetcher")

        # Test functionality - pass explicit process_name since test bypasses provisioner
        global_logger = await get_or_create_metric_logger(
            proc_mesh=procs, process_name="TestProcess"
        )

        # Get results to check
        proc_has_fetcher = hasattr(procs, "_local_fetcher")
        proc_id = procs._uid if hasattr(procs, "_uid") else None
        global_has_fetcher = (
            await global_logger.has_fetcher.call_one(proc_id) if proc_id else False
        )

        # Assert based on expected behavior
        if should_register_fetchers:
            assert (
                proc_has_fetcher
            ), f"Expected process to have _local_fetcher when FORGE_DISABLE_METRICS={env_var_value}"
            assert (
                global_has_fetcher
            ), f"Expected global logger to have fetcher registered when FORGE_DISABLE_METRICS={env_var_value}"
        else:
            assert (
                not proc_has_fetcher
            ), f"Expected process to NOT have _local_fetcher when FORGE_DISABLE_METRICS={env_var_value}"
            assert (
                not global_has_fetcher
            ), f"Expected global logger to NOT have fetcher registered when FORGE_DISABLE_METRICS={env_var_value}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "env_value,should_register",
        [
            ("false", True),
            ("true", False),
            (None, True),
        ],
    )
    async def test_fetcher_registration_with_env_flag(self, env_value, should_register):
        """Test fetcher registration behavior with different environment flag values."""
        await self._test_fetcher_registration(env_value, should_register)
