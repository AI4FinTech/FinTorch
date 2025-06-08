from fintorch.datasets.causal_data import CausalDataModule

if __name__ == "__main__":
    # Define parameters
    TIME_STEP = 16
    OUTPUT_WINDOW = 15
    BATCH_SIZE = 32
    # Use the current directory to find the data file
    DATA_PATH = "data_0.csv"
    DUMMY_DATA_ROWS = 200  # Number of rows for dummy data

    print("\nInitializing CausalDataModule...")
    try:
        # Instantiate the DataModule
        data_module = CausalDataModule(
            dataset_type='diamond',
            time_step=TIME_STEP,
            output_window=OUTPUT_WINDOW,
            batch_size=BATCH_SIZE,
            num_workers=0,  # Use 0 for simplicity in this script
            train_split=0.7,
            val_split=0.15,
        )

        # Setup the datamodule (loads data, creates splits)
        print("Setting up DataModule...")
        data_module.setup()
        print("Setup complete.")
        print(f"Total samples: {len(data_module.dataset)}")  # type: ignore
        print(f"Train samples: {len(data_module.train_dataset)}")  # type: ignore
        print(f"Val samples:   {len(data_module.val_dataset)}")  # type: ignore
        print(f"Test samples:  {len(data_module.test_dataset)}")  # type: ignore

        # --- Get Dataloaders and Print First Batch ---

        # Training DataLoader
        print("\n--- Training DataLoader ---")
        train_loader = data_module.train_dataloader()
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            train_data_batch = batch["past_target"]
            print(f"First Train Batch - Past Target Shape: {train_data_batch.shape}")
            print(f"First Train Batch - Output Target Shape: {batch['output_target'].shape}")
            print(f"First Train Batch - Static Real Shape: {batch['static_features_real'].shape}")
            print(
                "First element past data:\n", train_data_batch[0, :5, 0, 0]
            )  # Print first 5 steps of first series
            print("First element target:\n", batch['output_target'][0, :5, 0, 0])
        else:
            print("Training DataLoader is empty.")

        # Validation DataLoader
        print("\n--- Validation DataLoader ---")
        val_loader = data_module.val_dataloader()
        if len(val_loader) > 0:
            batch = next(iter(val_loader))
            val_data_batch = batch["past_target"]
            print(f"First Val Batch - Past Target Shape: {val_data_batch.shape}")
            print(f"First Val Batch - Output Target Shape: {batch['output_target'].shape}")
        else:
            print("Validation DataLoader is empty.")

        # Test DataLoader
        print("\n--- Test DataLoader ---")
        test_loader = data_module.test_dataloader()
        if len(test_loader) > 0:
            batch = next(iter(test_loader))
            test_data_batch = batch["past_target"]
            print(f"First Test Batch - Past Target Shape: {test_data_batch.shape}")
            print(f"First Test Batch - Output Target Shape: {batch['output_target'].shape}")
        else:
            print("Test DataLoader is empty.")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ValueError as val_error:
        print(f"Configuration error: {val_error}")
    except RuntimeError as rt_error:
        print(f"Runtime error: {rt_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
