import os
from typing import Optional

from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore


class KaggleDownloader:
    """
    A class for downloading datasets from Kaggle using the Kaggle API.

    Attributes:
        api (KaggleApi): The Kaggle API object used for authentication and downloading.
    """

    def __init__(self) -> None:
        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(
        self, dataset_name: str, download_dir: Optional[str] = None
    ) -> None:
        """
        Downloads the specified dataset from Kaggle.

        Args:
            dataset_name (str): The name of the dataset on Kaggle.
            download_dir (str, optional): The directory to save the downloaded files. If not provided,
                a default directory will be created.

        Returns:
            None
        """

        # TODO: check behavior in case of no default directory
        if download_dir is None:
            download_dir = os.path.join(os.getcwd(), "kaggledata")
            os.makedirs(download_dir, exist_ok=True)

        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        self.api.dataset_download_files(
            dataset_name, path=download_dir, unzip=True, quiet=False
        )
