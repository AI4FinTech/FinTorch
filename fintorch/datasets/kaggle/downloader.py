import os

from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleDownloader:
    """
    A class for downloading datasets from Kaggle using the Kaggle API.

    Attributes:
        api (KaggleApi): The Kaggle API object used for authentication and downloading.
    """

    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(self, dataset_name, download_dir=None):
        """
        Downloads the specified dataset from Kaggle.

        Args:
            dataset_name (str): The name of the dataset on Kaggle.
            download_dir (str, optional): The directory to save the downloaded files. If not provided,
                a default directory will be created.

        Returns:
            None
        """
        if download_dir is None:
            download_dir = self.create_fintorch_data_directory()
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        self.api.dataset_download_files(dataset_name,
                                        path=download_dir,
                                        unzip=True,
                                        quiet=False)
