import os

from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleDownloader:

    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(self, dataset_name, download_dir=None):
        if download_dir is None:
            download_dir = self.create_fintorch_data_directory()
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        self.api.dataset_download_files(dataset_name,
                                        path=download_dir,
                                        unzip=True,
                                        quiet=False)
