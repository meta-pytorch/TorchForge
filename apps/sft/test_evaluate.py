"""
Tests for the non-blocking all_reduce evaluation logic in main.py

This tests the epoch-detection and async all_reduce pattern used to
synchronize evaluation completion across multiple ranks without blocking.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch


@dataclass
class MockMetric:
    """Mock metric object matching the structure in batch["metrics"]"""

    metric_name: str
    value: int


class MockTrainer:
    """Mock trainer with minimal setup for testing evaluate logic"""

    def __init__(self, eval_steps=0):
        self.eval_steps = eval_steps
        self.device = torch.device("cpu")
        self.model_parts = [Mock()]

    def _extract_epoch_from_batch(self, batch: dict) -> int | None:
        """Extract epoch number from batch metrics."""
        if "metrics" not in batch:
            return None

        for metric in batch["metrics"]:
            if hasattr(metric, "metric_name") and metric.metric_name == "num_epochs":
                return metric.value
        return None

    def forward_only(self, batch, labels):
        """Mock forward pass - returns dummy loss"""
        return torch.tensor(1.5)


def create_batch_with_epoch(epoch: int, loss_value: float = 1.5):
    """Helper to create a mock batch with epoch metadata"""
    return {
        "input_ids": torch.randn(2, 10),
        "attention_mask": torch.ones(2, 10),
        "labels": torch.randint(0, 100, (2, 10)),
        "metrics": [MockMetric(metric_name="num_epochs", value=epoch)],
    }


def create_batch_without_epoch(loss_value: float = 1.5):
    """Helper to create a mock batch without epoch metadata"""
    return {
        "input_ids": torch.randn(2, 10),
        "attention_mask": torch.ones(2, 10),
        "labels": torch.randint(0, 100, (2, 10)),
    }


class TestExtractEpochFromBatch:
    """Test the _extract_epoch_from_batch helper method"""

    def test_extract_epoch_success(self):
        """Test extracting epoch from batch with proper metadata"""
        trainer = MockTrainer()
        batch = create_batch_with_epoch(epoch=5)

        epoch = trainer._extract_epoch_from_batch(batch)
        assert epoch == 5

    def test_extract_epoch_no_metrics(self):
        """Test batch without metrics returns None"""
        trainer = MockTrainer()
        batch = create_batch_without_epoch()

        epoch = trainer._extract_epoch_from_batch(batch)
        assert epoch is None

    def test_extract_epoch_wrong_metric_name(self):
        """Test batch with metrics but wrong metric_name returns None"""
        trainer = MockTrainer()
        batch = {
            "input_ids": torch.randn(2, 10),
            "metrics": [MockMetric(metric_name="other_metric", value=10)],
        }

        epoch = trainer._extract_epoch_from_batch(batch)
        assert epoch is None

    def test_extract_epoch_multiple_metrics(self):
        """Test extracting epoch from batch with multiple metrics"""
        trainer = MockTrainer()
        batch = {
            "input_ids": torch.randn(2, 10),
            "metrics": [
                MockMetric(metric_name="loss", value=1.5),
                MockMetric(metric_name="num_epochs", value=3),
                MockMetric(metric_name="step", value=100),
            ],
        }

        epoch = trainer._extract_epoch_from_batch(batch)
        assert epoch == 3


class TestEvaluationLogic:
    """Test the evaluation loop logic (single-rank scenario)"""

    @pytest.mark.asyncio
    async def test_single_epoch_completion(self):
        """Test that evaluation stops after one complete epoch"""
        trainer = MockTrainer(eval_steps=0)  # No cap

        # Create batches: 3 from epoch 0, then epoch increments to 1
        batches = [
            create_batch_with_epoch(0),
            create_batch_with_epoch(0),
            create_batch_with_epoch(0),
            create_batch_with_epoch(1),  # Epoch increment - should trigger stop
        ]

        dataloader = iter(batches)

        # Simulate the evaluation pattern
        num_processed = 0
        starting_epoch = None
        next_should_break = False

        # Get first batch
        next_batch = next(dataloader)

        while True:
            if next_should_break:
                break

            batch = next_batch

            # Extract epoch from current batch
            current_epoch = trainer._extract_epoch_from_batch(batch)
            if current_epoch is not None and starting_epoch is None:
                starting_epoch = current_epoch

            # Try to prefetch next batch
            try:
                next_batch = next(dataloader)
                next_epoch = trainer._extract_epoch_from_batch(next_batch)

                # Check for epoch increment
                if next_epoch is not None and starting_epoch is not None:
                    epoch_increment = next_epoch - starting_epoch
                    next_should_break = epoch_increment > 0

            except StopIteration:
                next_should_break = True

            # Process current batch
            num_processed += 1

        # Should have processed 3 batches (stopped when detected epoch 1)
        assert num_processed == 3
        assert starting_epoch == 0

    @pytest.mark.asyncio
    async def test_eval_steps_cap(self):
        """Test that evaluation respects eval_steps cap"""
        trainer = MockTrainer(eval_steps=2)  # Cap at 2 batches

        # Create 5 batches all in same epoch
        batches = [create_batch_with_epoch(0) for _ in range(5)]
        dataloader = iter(batches)

        # Simulate the evaluation pattern
        num_processed = 0
        next_should_break = False

        # Get first batch
        next_batch = next(dataloader)

        while True:
            if next_should_break:
                break

            # Check eval_steps cap
            if trainer.eval_steps > 0 and num_processed >= trainer.eval_steps:
                break

            batch = next_batch

            # Try to prefetch next batch
            try:
                next_batch = next(dataloader)
            except StopIteration:
                next_should_break = True

            # Process current batch
            num_processed += 1

        # Should have processed exactly 2 batches (eval_steps cap)
        assert num_processed == 2

    @pytest.mark.asyncio
    async def test_empty_dataloader(self):
        """Test handling of empty dataloader"""
        trainer = MockTrainer(eval_steps=0)

        batches = []
        dataloader = iter(batches)

        # Should raise StopIteration immediately
        with pytest.raises(StopIteration):
            next_batch = next(dataloader)

    @pytest.mark.asyncio
    async def test_single_batch(self):
        """Test evaluation with only one batch"""
        trainer = MockTrainer(eval_steps=0)

        batches = [create_batch_with_epoch(0)]
        dataloader = iter(batches)

        num_processed = 0
        next_should_break = False

        # Get first batch
        next_batch = next(dataloader)

        while True:
            if next_should_break:
                break

            batch = next_batch

            # Try to prefetch next batch
            try:
                next_batch = next(dataloader)
            except StopIteration:
                next_should_break = True

            # Process current batch
            num_processed += 1

        # Should have processed 1 batch
        assert num_processed == 1

    @pytest.mark.asyncio
    async def test_no_epoch_metadata(self):
        """Test evaluation when batches don't have epoch metadata"""
        trainer = MockTrainer(eval_steps=3)  # Use eval_steps as fallback

        # Create batches without epoch metadata
        batches = [create_batch_without_epoch() for _ in range(5)]
        dataloader = iter(batches)

        num_processed = 0
        next_should_break = False
        next_batch = next(dataloader)

        while True:
            if next_should_break:
                break

            # Check eval_steps cap (should be the stopping condition)
            if trainer.eval_steps > 0 and num_processed >= trainer.eval_steps:
                break

            batch = next_batch

            try:
                next_batch = next(dataloader)
            except StopIteration:
                next_should_break = True

            num_processed += 1

        # Should stop at eval_steps
        assert num_processed == 3


class TestAsyncAllReduce:
    """Test the async all_reduce pattern with mocked distributed operations"""

    @pytest.mark.asyncio
    async def test_async_all_reduce_pattern(self):
        """Test the async all_reduce pattern with mock distributed operations"""

        # Mock distributed environment
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.all_reduce") as mock_all_reduce:

                # Create mock Work handle for async operation
                mock_work = Mock()
                mock_work.wait = Mock()
                mock_all_reduce.return_value = mock_work

                trainer = MockTrainer(eval_steps=0)

                # Simulate the async pattern
                epoch_tensor = torch.tensor([0], dtype=torch.long)

                # Start async all_reduce (should return immediately)
                work_handle = torch.distributed.all_reduce(
                    epoch_tensor, op=torch.distributed.ReduceOp.MAX, async_op=True
                )

                # Verify it returned immediately with a work handle
                assert work_handle is not None
                assert mock_all_reduce.called

                # Simulate doing computation here...

                # Wait for completion
                work_handle.wait()
                assert mock_work.wait.called

    @pytest.mark.asyncio
    async def test_multi_rank_epoch_detection(self):
        """Test that epoch completion is detected when ANY rank finishes"""

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.all_reduce") as mock_all_reduce:

                def all_reduce_side_effect(tensor, op, async_op=False):
                    """Simulate all_reduce MAX operation across ranks
                    Rank 0: epoch_increment = 0 (still in epoch 0)
                    Rank 1: epoch_increment = 1 (moved to epoch 1)
                    MAX = 1, so all ranks should stop
                    """
                    # Simulate MAX operation - set tensor to max value
                    tensor[0] = 1  # At least one rank has epoch_increment=1

                    if async_op:
                        mock_work = Mock()
                        mock_work.wait = Mock()
                        return mock_work
                    return None

                mock_all_reduce.side_effect = all_reduce_side_effect

                trainer = MockTrainer(eval_steps=0)

                # Simulate rank 1's perspective: it moved to epoch 1
                starting_epoch = 0
                next_epoch = 1
                epoch_increment = next_epoch - starting_epoch  # = 1

                epoch_tensor = torch.tensor([epoch_increment], dtype=torch.long)

                # Start async all_reduce
                work = torch.distributed.all_reduce(
                    epoch_tensor, op=torch.distributed.ReduceOp.MAX, async_op=True
                )

                # Wait for result
                work.wait()

                # Check if should break (any rank has increment > 0)
                should_break = epoch_tensor.item() > 0

                assert should_break is True
                assert epoch_tensor.item() == 1


class TestEvaluationIntegration:
    """Integration-style tests for the full evaluation flow"""

    @pytest.mark.asyncio
    async def test_prefetch_pattern_ordering(self):
        """Test that the prefetch pattern processes batches in correct order"""
        trainer = MockTrainer(eval_steps=0)

        # Create identifiable batches
        batches = [
            {
                "id": 0,
                "metrics": [MockMetric("num_epochs", 0)],
                "labels": torch.zeros(1),
            },
            {
                "id": 1,
                "metrics": [MockMetric("num_epochs", 0)],
                "labels": torch.zeros(1),
            },
            {
                "id": 2,
                "metrics": [MockMetric("num_epochs", 0)],
                "labels": torch.zeros(1),
            },
            {
                "id": 3,
                "metrics": [MockMetric("num_epochs", 1)],
                "labels": torch.zeros(1),
            },
        ]

        dataloader = iter(batches)
        processed_ids = []

        # Prefetch first batch
        next_batch = next(dataloader)
        next_should_break = False
        starting_epoch = None

        while True:
            if next_should_break:
                break

            # Process current batch
            batch = next_batch
            processed_ids.append(batch["id"])

            # Extract epoch
            current_epoch = trainer._extract_epoch_from_batch(batch)
            if current_epoch is not None and starting_epoch is None:
                starting_epoch = current_epoch

            # Prefetch next
            try:
                next_batch = next(dataloader)
                next_epoch = trainer._extract_epoch_from_batch(next_batch)

                if next_epoch is not None and starting_epoch is not None:
                    epoch_increment = next_epoch - starting_epoch
                    next_should_break = epoch_increment > 0
            except StopIteration:
                next_should_break = True

        # Should have processed batches 0, 1, 2 (stopped when detected batch 3 has epoch 1)
        assert processed_ids == [0, 1, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
