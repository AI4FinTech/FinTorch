import os
import pytest
from fintorch.datasets.invoice import InvoiceDataset


@pytest.fixture
def invoice_dataset(tmp_path):
    return InvoiceDataset(root=tmp_path, force_reload=True)


def test_download(tmp_path):
    print(f"tmp path:{tmp_path}")
    # Create an instance of InvoiceDataset
    InvoiceDataset(root=tmp_path, force_reload=True)

    assert os.path.exists(
        os.path.join(tmp_path, "raw/dataset/training_data/annotations/")
    )
    assert os.path.exists(os.path.join(tmp_path, "raw/dataset/training_data/images/"))
    assert os.path.exists(os.path.join(tmp_path, "processed/training_data/"))
    assert os.path.exists(
        os.path.join(tmp_path, "raw/dataset/testing_data/annotations/")
    )
    assert os.path.exists(os.path.join(tmp_path, "raw/dataset/testing_data/images/"))
    assert os.path.exists(os.path.join(tmp_path, "processed/testing_data/"))

    training_data_files = os.listdir(os.path.join(tmp_path, "processed/training_data/"))
    testing_data_files = os.listdir(os.path.join(tmp_path, "processed/testing_data/"))

    assert len(training_data_files) > 0, "No files found in processed/training_data/"
    assert len(testing_data_files) > 0, "No files found in processed/testing_data/"
