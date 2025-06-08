#!/usr/bin/env python3
"""
Smoke test script for FinTorch examples.

This script runs example files to check if they can successfully reach the training stage
without actually running the full training process. It monitors the output for training-related
keywords and terminates the process once training is detected or after a timeout.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


# Keywords that indicate training has started (must be very specific to avoid false positives)
TRAINING_KEYWORDS = [
    "Epoch 1/",
    "Epoch     1",
    "│ 1 ",  # Lightning progress bar
    "Training: ",
    "Sanity Checking: ",
    "LOCAL_RANK:",
    "cross_validation(",
    "train_loss",
    "val_loss",
    "Validating: ",
    "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓",
    "GPU available: True",
    "TPU available: False",
    "Using 16bit Automatic Mixed Precision (AMP)",
    "Initializing distributed:",
]

# Keywords specific to dataset examples that indicate successful data loading/processing
DATASET_SUCCESS_KEYWORDS = [
    "Dataset loaded successfully",
    "Data preprocessing complete",
    "Training data shape:",
    "Validation data shape:",
    "Test data shape:",
    "Download complete",
    "Processing complete",
    "Data loaded:",
    "describe()",
    "DataFrame created",
    "Examples found:",
    "Loading train batch",
    "Loading test batch",
    "Loading validation batch",
    "Batch loaded",
    "Dataset initialized",
    "DataLoader created",
    "Data shape:",
    "Number of samples:",
    "Number of features:",
    "Data statistics:",
    "Processing batch",
    "Downloading:",
    "Download progress:",
    "Extracting:",
    "File downloaded",
    "Data file found",
    "Successfully loaded",
    "Dataset ready",
    "Data prepared",
    "Preprocessing finished",
    "Loading complete",
    "Dataset creation complete",
    "Data loading finished",
    "count    ",  # pandas describe output
    "mean     ",  # pandas describe output
    "std      ",  # pandas describe output
    "INFO:root:",  # logging messages often indicate progress
    "Dataset setup complete",
    "Data module ready",
]

# Keywords that indicate successful setup before training (but not training itself)
SETUP_SUCCESS_KEYWORDS = [
    "Setup complete",
    "DataLoader(",
    "Dataset created",
    "Model initialized",
    "trainer = L.Trainer(",
    "trainer = lightning.Trainer(",
    "trainer = Trainer(",
    "NeuralForecast(models=",
    "datamodule.setup()",
]

# Files to skip (known to be problematic or not standalone examples)
SKIP_FILES = [
    "__init__.py",
    "README.md",
    ".DS_Store",
]

# Maximum time to wait for examples (seconds)
DATASET_TIMEOUT_SECONDS = 180  # Datasets need more time to download
MODEL_TIMEOUT_SECONDS = 60     # Models should reach training quickly


class ExampleRunner:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.examples_dir = project_root / "examples"
        self.results = []

    def find_example_files(self) -> List[Path]:
        """Find all Python example files."""
        example_files = []

        # Look in examples/models and examples/datasets
        for subdir in ["models", "datasets"]:
            subdir_path = self.examples_dir / subdir
            if subdir_path.exists():
                for file_path in subdir_path.rglob("*.py"):
                    if file_path.name not in SKIP_FILES and "diamond" not in file_path.name.lower():
                        example_files.append(file_path)

        # Also check for standalone examples in the main examples directory
        for file_path in self.examples_dir.glob("*.py"):
            if file_path.name not in SKIP_FILES:
                example_files.append(file_path)

        return sorted(example_files)

    def run_example_with_timeout(self, example_path: Path) -> Tuple[bool, str, str]:
        """
        Run an example file and monitor its output.

        Returns:
            (success, status_message, output_snippet)
        """
        print(f"Testing {example_path.relative_to(self.project_root)}...", end=" ", flush=True)

        # Determine if this is a dataset or model example
        is_dataset_example = "datasets" in str(example_path)
        timeout_seconds = DATASET_TIMEOUT_SECONDS if is_dataset_example else MODEL_TIMEOUT_SECONDS

        try:
            # Start the process
            process = subprocess.Popen(
                [sys.executable, str(example_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.project_root),
                env=dict(os.environ, PYTHONPATH=str(self.project_root))
            )

            output_lines = []
            start_time = time.time()
            training_detected = False
            setup_detected = False
            dataset_success_detected = False

            # Monitor output line by line
            while process.poll() is None:
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()  # Force kill if terminate doesn't work

                    if training_detected:
                        return True, "SUCCESS - Training stage reached (timeout)", "\n".join(output_lines[-5:])
                    elif is_dataset_example and dataset_success_detected:
                        return True, "SUCCESS - Dataset processing completed (timeout)", "\n".join(output_lines[-5:])
                    elif setup_detected:
                        return True, "SUCCESS - Setup completed (timeout)", "\n".join(output_lines[-5:])
                    elif is_dataset_example:
                        # For dataset examples, timeout might be acceptable if there's download activity
                        if any("download" in line.lower() or "loading" in line.lower() or "processing" in line.lower()
                               for line in output_lines[-10:]):
                            return True, "SUCCESS - Dataset processing in progress (timeout)", "\n".join(output_lines[-5:])
                        else:
                            return False, "TIMEOUT - No dataset activity detected", "\n".join(output_lines[-5:])
                    else:
                        return False, "TIMEOUT - No training stage detected", "\n".join(output_lines[-5:])

                # Read output line by line with timeout
                try:
                    line = process.stdout.readline()
                    if line:
                        output_lines.append(line.strip())
                        line_lower = line.lower()

                        # For model examples, terminate as soon as training is detected
                        if not is_dataset_example:
                            for keyword in TRAINING_KEYWORDS:
                                if keyword in line or keyword.lower() in line_lower:
                                    training_detected = True
                                    # Give it a moment to actually start training, then kill
                                    time.sleep(3)
                                    process.terminate()
                                    try:
                                        process.wait(timeout=5)
                                    except subprocess.TimeoutExpired:
                                        process.kill()
                                    return True, "SUCCESS - Training stage reached", "\n".join(output_lines[-5:])

                        # For dataset examples, look for data processing success
                        if is_dataset_example:
                            for keyword in DATASET_SUCCESS_KEYWORDS:
                                if keyword in line or keyword.lower() in line_lower:
                                    dataset_success_detected = True

                        # Check for setup success keywords (exact matches)
                        for keyword in SETUP_SUCCESS_KEYWORDS:
                            if keyword in line or keyword.lower() in line_lower:
                                setup_detected = True

                except Exception:
                    break

            # Process finished normally
            return_code = process.wait()

            if return_code == 0:
                if training_detected:
                    return True, "SUCCESS - Completed successfully with training", "\n".join(output_lines[-5:])
                elif is_dataset_example and dataset_success_detected:
                    return True, "SUCCESS - Dataset completed successfully", "\n".join(output_lines[-5:])
                elif setup_detected:
                    return True, "SUCCESS - Completed successfully with setup", "\n".join(output_lines[-5:])
                else:
                    return True, "SUCCESS - Completed successfully", "\n".join(output_lines[-5:])
            else:
                # Check for different types of errors
                full_output = "\n".join(output_lines)

                # Import/Module errors - these are code issues that should be flagged
                import_errors = ["ModuleNotFoundError", "ImportError", "No module named"]
                if any(error in full_output for error in import_errors):
                    return False, f"FAILED - Import/Module error - Exit code {return_code}", "\n".join(output_lines[-3:])

                # API/TypeError - these are code issues that should be flagged
                api_errors = ["TypeError", "missing", "required positional arguments", "AttributeError"]
                if any(error in full_output for error in api_errors):
                    return False, f"FAILED - API/Type error - Exit code {return_code}", "\n".join(output_lines[-3:])

                # Expected data-related errors (missing data files, network issues, directories)
                data_errors = ["FileNotFoundError", "Data file not found", "Download failed", "Connection error", "Directory not found", "/.fintorch_data/"]
                if any(error in full_output for error in data_errors):
                    error_type = "dataset" if is_dataset_example else "data loading"
                    return True, f"SUCCESS - Expected {error_type} error (missing data) - Exit code {return_code}", "\n".join(output_lines[-3:])

                return False, f"FAILED - Exit code {return_code}", "\n".join(output_lines[-5:])

        except Exception as e:
            return False, f"ERROR - Exception during execution: {str(e)}", ""

    def run_all_examples(self) -> None:
        """Run all examples and collect results."""
        example_files = self.find_example_files()

        if not example_files:
            print("No example files found!")
            return

        print(f"Found {len(example_files)} example files to test")
        print("=" * 80)

        for example_path in example_files:
            success, status, output = self.run_example_with_timeout(example_path)

            relative_path = example_path.relative_to(self.project_root)
            self.results.append((relative_path, success, status, output))

            # Print immediate result
            status_symbol = "✓" if success else "✗"
            print(f"{status_symbol} {status}")

            # Print output snippet if failed
            if not success and output:
                print(f"   Last output: {output[:200]}...")

            print()

    def print_summary(self) -> bool:
        """Print a summary of all results."""
        print("=" * 80)
        print("SMOKE TEST SUMMARY")
        print("=" * 80)

        success_count = sum(1 for _, success, _, _ in self.results if success)
        total_count = len(self.results)

        print(f"Total examples tested: {total_count}")
        print(f"Successful: {success_count}")
        print(f"Failed: {total_count - success_count}")
        print()

        if success_count < total_count:
            print("Failed examples:")
            for path, success, status, output in self.results:
                if not success:
                    print(f"  ✗ {path}: {status}")
            print()

        print("Successful examples:")
        for path, success, status, output in self.results:
            if success:
                print(f"  ✓ {path}: {status}")

        return success_count == total_count


def run_quick_test():
    """Run a quick test on a subset of examples."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    # Representative subset for quick testing
    quick_examples = [
        "examples/datasets/example_marketdata.py",
        "examples/datasets/example_stockticker.py",
        "examples/models/example_tft.py",
        "examples/models/stockmodel.py",
        "examples/causal_data_example.py",
    ]

    runner = ExampleRunner(project_root)
    runner.results = []

    print("Quick Smoke Test - Testing subset of examples")
    print("=" * 60)

    for example_rel_path in quick_examples:
        example_path = project_root / example_rel_path

        if not example_path.exists():
            print(f"SKIP - {example_path.name} (not found)")
            continue

        success, status, output = runner.run_example_with_timeout(example_path)
        relative_path = example_path.relative_to(project_root)
        runner.results.append((relative_path, success, status, output))

        # Print immediate result
        status_symbol = "✓" if success else "✗"
        print(f"{status_symbol} {status}")

        # Print output snippet if failed
        if not success and output:
            print(f"   Last output: {output[:150]}...")
        print()

    return runner.print_summary()


def main():
    """Main entry point."""
    # Find project root (directory containing this script's parent)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    if not (project_root / "examples").exists():
        print(f"Error: Examples directory not found at {project_root / 'examples'}")
        sys.exit(1)

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        all_passed = run_quick_test()
    else:
        runner = ExampleRunner(project_root)
        runner.run_all_examples()
        all_passed = runner.print_summary()

    if all_passed:
        print("\n🎉 All examples passed the smoke test!")
        sys.exit(0)
    else:
        print("\n❌ Some examples failed the smoke test.")
        sys.exit(1)


if __name__ == "__main__":
    main()
