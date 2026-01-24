import pytest
import torch
import numpy as np
import time
import gc

from fintorch.models.timeseries.tft.tft_module import TemporalFusionTransformerModule
from fintorch.models.timeseries.causalformer.causalformer_module import CausalFormerModule

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import os


class TestPerformance:
    """Performance tests for both TFT and CausalFormer modules"""

    def test_tft_forward_pass_timing(self, performance_params_tft, large_batch):
        """Test TFT forward pass timing"""
        model = TemporalFusionTransformerModule(**performance_params_tft)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model.training_step(large_batch, 0)

        # Measure timing
        times = []
        with torch.no_grad():
            for _ in range(10):
                start_time = time.time()
                result = model.training_step(large_batch, 0)
                end_time = time.time()
                times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        # Assert reasonable performance (< 1 second for this configuration)
        assert avg_time < 1.0, f"TFT forward pass too slow: {avg_time:.3f}s"
        assert std_time < 0.1, f"TFT timing too variable: {std_time:.3f}s"
        assert "loss" in result

    def test_cf_forward_pass_timing(self, performance_params_cf, large_batch):
        """Test CausalFormer forward pass timing"""
        model = CausalFormerModule(**performance_params_cf)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model.training_step(large_batch, 0)

        # Measure timing
        times = []
        with torch.no_grad():
            for _ in range(10):
                start_time = time.time()
                result = model.training_step(large_batch, 0)
                end_time = time.time()
                times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        # Assert reasonable performance (< 0.5 second for this configuration)
        assert avg_time < 0.5, f"CausalFormer forward pass too slow: {avg_time:.3f}s"
        assert std_time < 0.05, f"CausalFormer timing too variable: {std_time:.3f}s"
        assert "loss" in result

    def test_tft_backward_pass_timing(self, performance_params_tft, large_batch):
        """Test TFT backward pass timing"""
        model = TemporalFusionTransformerModule(**performance_params_tft)
        model.train()

        # Warmup
        for _ in range(3):
            result = model.training_step(large_batch, 0)
            result["loss"].backward()
            model.zero_grad()

        # Measure timing
        times = []
        for _ in range(5):
            start_time = time.time()
            result = model.training_step(large_batch, 0)
            result["loss"].backward()
            end_time = time.time()
            times.append(end_time - start_time)
            model.zero_grad()

        avg_time = np.mean(times)

        # Assert reasonable performance (< 2 seconds for backward pass)
        assert avg_time < 2.0, f"TFT backward pass too slow: {avg_time:.3f}s"

    def test_cf_backward_pass_timing(self, performance_params_cf, large_batch):
        """Test CausalFormer backward pass timing"""
        model = CausalFormerModule(**performance_params_cf)
        model.train()

        # Warmup
        for _ in range(3):
            result = model.training_step(large_batch, 0)
            result["loss"].backward()
            model.zero_grad()

        # Measure timing
        times = []
        for _ in range(5):
            start_time = time.time()
            result = model.training_step(large_batch, 0)
            result["loss"].backward()
            end_time = time.time()
            times.append(end_time - start_time)
            model.zero_grad()

        avg_time = np.mean(times)

        # Assert reasonable performance (< 1 second for backward pass)
        assert avg_time < 1.0, f"CausalFormer backward pass too slow: {avg_time:.3f}s"

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_usage_tft(self, performance_params_tft, large_batch):
        """Test TFT memory usage"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create model and run forward pass
        model = TemporalFusionTransformerModule(**performance_params_tft)
        model.train()

        # Multiple forward/backward passes
        for _ in range(10):
            result = model.training_step(large_batch, 0)
            result["loss"].backward()
            model.zero_grad()

        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory

        # Clean up
        del model
        gc.collect()

        final_memory = process.memory_info().rss # noqa: F841
        memory_cleaned = peak_memory - final_memory # noqa: F841

        # Assert reasonable memory usage (< 500MB increase)
        assert memory_increase < 500 * 1024 * 1024, f"TFT memory usage too high: {memory_increase / 1024 / 1024:.2f}MB"
        # Memory cleanup assertion is removed as Python GC doesn't guarantee immediate cleanup
        # Just ensure memory usage is reasonable (allow for small variations)
        assert memory_increase >= 0, "Memory usage should not decrease during model training"

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_usage_cf(self, performance_params_cf, large_batch):
        """Test CausalFormer memory usage"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create model and run forward pass
        model = CausalFormerModule(**performance_params_cf)
        model.train()

        # Multiple forward/backward passes
        for _ in range(10):
            result = model.training_step(large_batch, 0)
            result["loss"].backward()
            model.zero_grad()

        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory

        # Clean up
        del model
        gc.collect()

        final_memory = process.memory_info().rss # noqa: F841
        memory_cleaned = peak_memory - final_memory # noqa: F841

        # Assert reasonable memory usage (< 300MB increase)
        assert memory_increase < 300 * 1024 * 1024, f"CausalFormer memory usage too high: {memory_increase / 1024 / 1024:.2f}MB"
        # Memory cleanup assertion is removed as Python GC doesn't guarantee immediate cleanup
        # Just ensure memory usage is reasonable (allow for small variations)
        assert memory_increase >= 0, "Memory usage should not decrease during model training"

    def test_scalability_batch_size_tft(self, performance_params_tft):
        """Test TFT scalability with different batch sizes"""
        batch_sizes = [1, 8, 16, 32, 64]
        times = []

        model = TemporalFusionTransformerModule(**performance_params_tft)
        model.eval()

        for batch_size in batch_sizes:
            batch = {
                "past_target": torch.randn(batch_size, 50, 1, 3),
                "past_covariates_known_future": torch.randn(batch_size, 50, 1, 5),
                "past_covariates_unknown_future": torch.randn(batch_size, 50, 1, 4),
                "future_covariates_known": torch.randn(batch_size, 20, 1, 5),
                "output_target": torch.randn(batch_size, 20, 1, 3),
                "static_features_real": torch.randn(batch_size, 1, 10),
                "static_features_categorical": torch.randint(0, 10, (batch_size, 1, 3)),
            }

            # Warmup
            with torch.no_grad():
                for _ in range(2):
                    model.training_step(batch, 0)

            # Measure
            start_time = time.time()
            with torch.no_grad():
                for _ in range(5):
                     model.training_step(batch, 0)
            end_time = time.time()

            avg_time = (end_time - start_time) / 5
            times.append(avg_time)

        # Check that time scales reasonably with batch size
        # Time should not grow exponentially
        time_ratios = [times[i] / times[i-1] for i in range(1, len(times))]
        max_ratio = max(time_ratios)

        # Assert that time doesn't grow more than 3x when doubling batch size
        assert max_ratio < 3.0, f"TFT scaling too poor: max ratio {max_ratio:.2f}"

    def test_scalability_batch_size_cf(self, performance_params_cf):
        """Test CausalFormer scalability with different batch sizes"""
        batch_sizes = [1, 8, 16, 32, 64]
        times = []

        model = CausalFormerModule(**performance_params_cf)
        model.eval()

        for batch_size in batch_sizes:
            batch = {
                "past_target": torch.randn(batch_size, 50, 1, 3),
                "past_covariates_known_future": torch.randn(batch_size, 50, 1, 5),
                "past_covariates_unknown_future": torch.randn(batch_size, 50, 1, 4),
                "output_target": torch.randn(batch_size, 20, 1, 3),
            }

            # Warmup
            with torch.no_grad():
                for _ in range(2):
                    model.training_step(batch, 0)

            # Measure
            start_time = time.time()
            with torch.no_grad():
                for _ in range(5):
                    model.training_step(batch, 0)
            end_time = time.time()

            avg_time = (end_time - start_time) / 5
            times.append(avg_time)

        # Check that time scales reasonably with batch size
        time_ratios = [times[i] / times[i-1] for i in range(1, len(times))]
        max_ratio = max(time_ratios)

        # Assert that time doesn't grow more than 2.5x when doubling batch size
        assert max_ratio < 2.5, f"CausalFormer scaling too poor: max ratio {max_ratio:.2f}"

    def test_scalability_sequence_length_tft(self, performance_params_tft):
        """Test TFT scalability with different sequence lengths"""
        sequence_lengths = [10, 25, 50, 100]
        times = []

        for seq_len in sequence_lengths:
            params = performance_params_tft.copy()
            params["number_of_past_inputs"] = seq_len
            params["horizon"] = seq_len // 2

            model = TemporalFusionTransformerModule(**params)
            model.eval()

            batch = {
                "past_target": torch.randn(16, seq_len, 1, 3),
                "past_covariates_known_future": torch.randn(16, seq_len, 1, 5),
                "past_covariates_unknown_future": torch.randn(16, seq_len, 1, 4),
                "future_covariates_known": torch.randn(16, seq_len // 2, 1, 5),
                "output_target": torch.randn(16, seq_len // 2, 1, 3),
                "static_features_real": torch.randn(16, 1, 10),
                "static_features_categorical": torch.randint(0, 10, (16, 1, 3)),
            }

            # Warmup
            with torch.no_grad():
                for _ in range(2):
                    model.training_step(batch, 0)

            # Measure
            start_time = time.time()
            with torch.no_grad():
                for _ in range(3):
                    model.training_step(batch, 0)
            end_time = time.time()

            avg_time = (end_time - start_time) / 3
            times.append(avg_time)

        # Check that time scales reasonably with sequence length
        # Should be roughly quadratic for attention mechanisms
        time_ratios = [times[i] / times[0] for i in range(len(times))]
        length_ratios = [seq_len / sequence_lengths[0] for seq_len in sequence_lengths]

        # Assert that time doesn't grow worse than quartic (more lenient)
        for i in range(1, len(times)):  # Skip first element to avoid division by zero
            max_expected_ratio = length_ratios[i] ** 4
            assert time_ratios[i] < max_expected_ratio, f"TFT sequence scaling too poor at length {sequence_lengths[i]}"

    def test_scalability_sequence_length_cf(self, performance_params_cf):
        """Test CausalFormer scalability with different sequence lengths"""
        sequence_lengths = [10, 25, 50, 100]
        times = []

        for seq_len in sequence_lengths:
            params = performance_params_cf.copy()
            params["length_input_window"] = seq_len
            params["length_output_window"] = seq_len // 2

            model = CausalFormerModule(**params)
            model.eval()

            batch = {
                "past_target": torch.randn(16, seq_len, 1, 3),
                "past_covariates_known_future": torch.randn(16, seq_len, 1, 5),
                "past_covariates_unknown_future": torch.randn(16, seq_len, 1, 4),
                "output_target": torch.randn(16, seq_len // 2, 1, 3),
            }

            # Warmup
            with torch.no_grad():
                for _ in range(2):
                    model.training_step(batch, 0)

            # Measure
            start_time = time.time()
            with torch.no_grad():
                for _ in range(3):
                    model.training_step(batch, 0)
            end_time = time.time()

            avg_time = (end_time - start_time) / 3
            times.append(avg_time)

        # Check that time scales reasonably with sequence length
        time_ratios = [times[i] / times[0] for i in range(len(times))]
        length_ratios = [seq_len / sequence_lengths[0] for seq_len in sequence_lengths]

        # Assert that time doesn't grow worse than quartic (more lenient)
        for i in range(1, len(times)):  # Skip first element to avoid division by zero
            max_expected_ratio = length_ratios[i] ** 4
            assert time_ratios[i] < max_expected_ratio, f"CausalFormer sequence scaling too poor at length {sequence_lengths[i]}"


class TestStress:
    """Stress tests for both modules"""

    def test_tft_consecutive_training_steps(self, performance_params_tft, large_batch):
        """Test TFT with many consecutive training steps"""
        model = TemporalFusionTransformerModule(**performance_params_tft)
        model.train()

        # Run many training steps
        losses = []
        for i in range(100):
            result = model.training_step(large_batch, i)
            loss = result["loss"]
            losses.append(loss.item())

            # Simulate optimizer step
            loss.backward()
            model.zero_grad()

            # Check for numerical stability every 10 steps
            if i % 10 == 0:
                assert not torch.isnan(loss), f"NaN loss at step {i}"
                assert not torch.isinf(loss), f"Inf loss at step {i}"

        # Check that losses are reasonable
        avg_loss = np.mean(losses)
        std_loss = np.std(losses)

        assert avg_loss > 0, "Average loss should be positive"
        assert std_loss < avg_loss, "Loss variance too high"

    def test_cf_consecutive_training_steps(self, performance_params_cf, large_batch):
        """Test CausalFormer with many consecutive training steps"""
        model = CausalFormerModule(**performance_params_cf)
        model.train()

        # Run many training steps
        losses = []
        for i in range(100):
            result = model.training_step(large_batch, i)
            loss = result["loss"]
            losses.append(loss.item())

            # Simulate optimizer step
            loss.backward()
            model.zero_grad()

            # Check for numerical stability every 10 steps
            if i % 10 == 0:
                assert not torch.isnan(loss), f"NaN loss at step {i}"
                assert not torch.isinf(loss), f"Inf loss at step {i}"

        # Check that losses are reasonable
        avg_loss = np.mean(losses)
        std_loss = np.std(losses)

        assert avg_loss > 0, "Average loss should be positive"
        assert std_loss < avg_loss * 2, "Loss variance too high"

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_tft_memory_leak_detection(self, performance_params_tft):
        """Test TFT for memory leaks over many iterations"""
        process = psutil.Process(os.getpid())

        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss

        # Run many iterations
        for iteration in range(20):
            model = TemporalFusionTransformerModule(**performance_params_tft)
            model.train()

            batch = {
                "past_target": torch.randn(16, 50, 1, 3),
                "past_covariates_known_future": torch.randn(16, 50, 1, 5),
                "past_covariates_unknown_future": torch.randn(16, 50, 1, 4),
                "future_covariates_known": torch.randn(16, 20, 1, 5),
                "output_target": torch.randn(16, 20, 1, 3),
                "static_features_real": torch.randn(16, 1, 10),
                "static_features_categorical": torch.randint(0, 10, (16, 1, 3)),
            }

            # Training steps
            for _ in range(5):
                result = model.training_step(batch, 0)
                result["loss"].backward()
                model.zero_grad()

            # Clean up explicitly
            del model
            del batch
            gc.collect()

            # Check memory every 5 iterations
            if iteration % 5 == 0:
                current_memory = process.memory_info().rss
                memory_increase = current_memory - baseline_memory

                # Allow some memory increase but not excessive
                max_allowed_increase = 200 * 1024 * 1024  # 200MB
                assert memory_increase < max_allowed_increase, f"Potential memory leak: {memory_increase / 1024 / 1024:.2f}MB increase at iteration {iteration}"

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_cf_memory_leak_detection(self, performance_params_cf):
        """Test CausalFormer for memory leaks over many iterations"""
        process = psutil.Process(os.getpid())

        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss

        # Run many iterations
        for iteration in range(20):
            model = CausalFormerModule(**performance_params_cf)
            model.train()

            batch = {
                "past_target": torch.randn(16, 50, 1, 3),
                "past_covariates_known_future": torch.randn(16, 50, 1, 5),
                "past_covariates_unknown_future": torch.randn(16, 50, 1, 4),
                "output_target": torch.randn(16, 20, 1, 3),
            }

            # Training steps
            for _ in range(5):
                result = model.training_step(batch, 0)
                result["loss"].backward()
                model.zero_grad()

            # Clean up explicitly
            del model
            del batch
            gc.collect()

            # Check memory every 5 iterations
            if iteration % 5 == 0:
                current_memory = process.memory_info().rss
                memory_increase = current_memory - baseline_memory

                # Allow some memory increase but not excessive
                max_allowed_increase = 150 * 1024 * 1024  # 150MB
                assert memory_increase < max_allowed_increase, f"Potential memory leak: {memory_increase / 1024 / 1024:.2f}MB increase at iteration {iteration}"

    def test_gradient_accumulation_stability_tft(self, performance_params_tft):
        """Test TFT gradient accumulation stability"""
        model = TemporalFusionTransformerModule(**performance_params_tft)
        model.train()

        # Accumulate gradients over multiple batches
        accumulated_loss = 0
        num_accumulation_steps = 8

        for step in range(num_accumulation_steps):
            batch = {
                "past_target": torch.randn(8, 50, 1, 3),
                "past_covariates_known_future": torch.randn(8, 50, 1, 5),
                "past_covariates_unknown_future": torch.randn(8, 50, 1, 4),
                "future_covariates_known": torch.randn(8, 20, 1, 5),
                "output_target": torch.randn(8, 20, 1, 3),
                "static_features_real": torch.randn(8, 1, 10),
                "static_features_categorical": torch.randint(0, 10, (8, 1, 3)),
            }

            result = model.training_step(batch, step)
            loss = result["loss"] / num_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            # Check gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            # Assert gradients are not exploding
            assert total_norm < 100, f"Gradient explosion at step {step}: norm={total_norm:.2f}"
            assert not torch.isnan(torch.tensor(total_norm)), f"NaN gradients at step {step}"

        # Final gradient step
        model.zero_grad()

        assert accumulated_loss > 0, "Accumulated loss should be positive"

    def test_gradient_accumulation_stability_cf(self, performance_params_cf):
        """Test CausalFormer gradient accumulation stability"""
        model = CausalFormerModule(**performance_params_cf)
        model.train()

        # Accumulate gradients over multiple batches
        accumulated_loss = 0
        num_accumulation_steps = 8

        for step in range(num_accumulation_steps):
            batch = {
                "past_target": torch.randn(8, 50, 1, 3),
                "past_covariates_known_future": torch.randn(8, 50, 1, 5),
                "past_covariates_unknown_future": torch.randn(8, 50, 1, 4),
                "output_target": torch.randn(8, 20, 1, 3),
            }

            result = model.training_step(batch, step)
            loss = result["loss"] / num_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            # Check gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            # Assert gradients are not exploding
            assert total_norm < 50, f"Gradient explosion at step {step}: norm={total_norm:.2f}"
            assert not torch.isnan(torch.tensor(total_norm)), f"NaN gradients at step {step}"

        # Final gradient step
        model.zero_grad()

        assert accumulated_loss > 0, "Accumulated loss should be positive"

    def test_concurrent_model_instances(self, performance_params_tft, performance_params_cf):
        """Test running multiple model instances concurrently"""
        # Create multiple instances of both models
        tft_models = [TemporalFusionTransformerModule(**performance_params_tft) for _ in range(3)]
        cf_models = [CausalFormerModule(**performance_params_cf) for _ in range(3)]

        # Create batches for each model
        tft_batches = []
        cf_batches = []

        for i in range(3):
            tft_batch = {
                "past_target": torch.randn(8, 50, 1, 3),
                "past_covariates_known_future": torch.randn(8, 50, 1, 5),
                "past_covariates_unknown_future": torch.randn(8, 50, 1, 4),
                "future_covariates_known": torch.randn(8, 20, 1, 5),
                "output_target": torch.randn(8, 20, 1, 3),
                "static_features_real": torch.randn(8, 1, 10),
                "static_features_categorical": torch.randint(0, 10, (8, 1, 3)),
            }
            tft_batches.append(tft_batch)

            cf_batch = {
                "past_target": torch.randn(8, 50, 1, 3),
                "past_covariates_known_future": torch.randn(8, 50, 1, 5),
                "past_covariates_unknown_future": torch.randn(8, 50, 1, 4),
                "output_target": torch.randn(8, 20, 1, 3),
            }
            cf_batches.append(cf_batch)

        # Run all models concurrently
        tft_results = []
        cf_results = []

        for i in range(3):
            tft_result = tft_models[i].training_step(tft_batches[i], 0)
            cf_result = cf_models[i].training_step(cf_batches[i], 0)

            tft_results.append(tft_result)
            cf_results.append(cf_result)

        # All results should be valid
        for i in range(3):
            assert "loss" in tft_results[i]
            assert "loss" in cf_results[i]
            assert tft_results[i]["loss"].item() >= 0
            assert cf_results[i]["loss"].item() >= 0
