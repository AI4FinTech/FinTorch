import logging
from typing import Any, List, Dict
import os
import json

from torch.utils.data import Dataset
import torchvision.transforms as transforms  # type: ignore

import torch

from PIL import Image

import requests
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm


class InvoiceDataset(Dataset):  # type: ignore
    def __init__(self, root: str, split: str = "train", force_reload: bool = False):
        super().__init__()
        self.root = root
        self.data: List[Any] = []  # Initialize data attribute

        if split not in ["train", "test", "all"]:
            raise ValueError("split must be one of: train, test, all")
        self.split = split

        logging.info("Loading invoice dataset")
        self.setupDirectories()

        if force_reload or not all(
            os.path.exists(path) and os.listdir(path) for path in self.processed_paths()
        ):
            # if we want to force reload, or a processed file is missing or empty. Start the processing
            self.download()  # download auction data
            self.process()

        self.load()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]

    def processed_paths(self) -> List[str]:
        return [
            os.path.join(self.root, path)
            for path in [
                "processed/training_data/",
                "processed/testing_data",
            ]
        ]

    def setupDirectories(self) -> None:
        try:
            os.makedirs(self.root, exist_ok=True)
            os.makedirs(os.path.join(self.root, "raw"), exist_ok=True)
            os.makedirs(os.path.join(self.root, "processed"), exist_ok=True)
            os.makedirs(
                os.path.join(self.root, "processed/training_data"), exist_ok=True
            )
            os.makedirs(
                os.path.join(self.root, "processed/testing_data"), exist_ok=True
            )
        except OSError as e:
            logging.error(f"Failed to create directories: {str(e)}")
            raise RuntimeError(f"Failed to setup directories: {str(e)}") from e

    def process(self) -> None:
        logging.info("Processing: apply transformation to FUNSD dataset")

        logging.info("Processing training data")
        self.process_dir(
            os.path.join(self.root, "raw/dataset/training_data/annotations/"),
            os.path.join(self.root, "raw/dataset/training_data/images/"),
            os.path.join(self.root, "processed/training_data/"),
        )

        logging.info("Processing test data")
        self.process_dir(
            os.path.join(self.root, "raw/dataset/testing_data/annotations/"),
            os.path.join(self.root, "raw/dataset/testing_data/images/"),
            os.path.join(self.root, "processed/testing_data/"),
        )

    def process_dir(self, annotation_dir: str, image_dir: str, target_dir: str) -> None:
        for file in tqdm(os.listdir(annotation_dir), desc="Processing files"):
            # Read the corresponding image file

            try:
                with open(os.path.join(annotation_dir, file), "r") as f:
                    data = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logging.error(f"Failed to read annotation file {file}: {str(e)}")
                continue

            image_file = os.path.splitext(file)[0] + ".png"
            image_path = os.path.join(image_dir, image_file)

            try:
                image = Image.open(image_path).convert("RGB")
            except IOError as e:
                logging.error(f"Failed to open image file {image_file}: {str(e)}")
                continue

            transform = transforms.Compose([transforms.ToTensor()])
            tensor_image = transform(image)

            data_dict = {"image": tensor_image, "meta": data["form"]}

            target_file = os.path.join(target_dir, os.path.splitext(file)[0] + ".pt")
            torch.save(data_dict, target_file)

    def load(self) -> None:
        if self.split == "train":
            self.data = self.load_dir(
                os.path.join(self.root, "processed/training_data/")
            )
        elif self.split == "test":
            self.data = self.load_dir(
                os.path.join(self.root, "processed/testing_data/")
            )
        else:  # all
            self.data = self.load_dir(
                os.path.join(self.root, "processed/training_data/")
            )
            self.data.extend(
                self.load_dir(os.path.join(self.root, "processed/testing_data/"))
            )

    def load_dir(self, dir: str) -> List[Dict[str, Any]]:
        data = []
        for file in os.listdir(dir):
            if file.endswith(".pt"):
                file_path = os.path.join(dir, file)
                try:
                    data.append(torch.load(file_path))
                except Exception as e:
                    logging.error(f"Failed to load {file_path}: {str(e)}")
                    continue

        return data

    def download(self) -> None:
        logging.info("Downloading the FUNSD dataset")
        url = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            logging.error(
                f"Failed to download dataset: HTTP status code {response.status_code}"
            )
            raise Exception(
                f"Failed to download dataset: HTTP status code {response.status_code}"
            )

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        tqdm_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

        with BytesIO() as byte_stream:
            for data in response.iter_content(block_size):
                tqdm_bar.update(len(data))
                byte_stream.write(data)
            tqdm_bar.close()
            byte_stream.seek(0)
            with ZipFile(byte_stream) as zip_ref:
                zip_ref.extractall(os.path.join(self.root, "raw/"))
                logging.info("Download and extraction complete")
