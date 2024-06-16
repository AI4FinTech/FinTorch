from typing import Callable, Optional

import lightning.pytorch as pl
from huggingface_hub import hf_hub_download
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm


class StockTicker(InMemoryDataset):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        super().__init__(root,
                         transform,
                         pre_transform,
                         pre_filter,
                         force_reload=force_reload)

    @property
    def raw_file_names(self):
        return [
            "BETA10.csv", "BETA20.csv", "BETA30.csv", "BETA5.csv",
            "BETA60.csv", "CNTD10.csv", "CNTD20.csv", "CNTD30.csv",
            "CNTD5.csv", "CNTD60.csv", "CNTN10.csv", "CNTN20.csv",
            "CNTN30.csv", "CNTN5.csv", "CNTN60.csv", "CNTP10.csv",
            "CNTP20.csv", "CNTP30.csv", "CNTP5.csv", "CNTP60.csv",
            "CORD10.csv", "CORD20.csv", "CORD30.csv", "CORD5.csv",
            "CORD60.csv", "CORR10.csv", "CORR20.csv", "CORR30.csv",
            "CORR5.csv", "CORR60.csv", "HIGH0.csv", "IMAX10.csv", "IMAX20.csv",
            "IMAX30.csv", "IMAX5.csv", "IMAX60.csv", "IMIN10.csv",
            "IMIN20.csv", "IMIN30.csv", "IMIN5.csv", "IMIN60.csv",
            "IMXD10.csv", "IMXD20.csv", "IMXD30.csv", "IMXD5.csv",
            "IMXD60.csv", "KLEN.csv", "KLOW.csv", "KLOW2.csv", "KMID.csv",
            "KMID2.csv", "KSFT.csv", "KSFT2.csv", "KUP.csv", "KUP2.csv",
            "LABEL0.csv", "LOW0.csv", "MA10.csv", "MA20.csv", "MA30.csv",
            "MA5.csv", "MA60.csv", "MAX10.csv", "MAX20.csv", "MAX30.csv",
            "MAX5.csv", "MAX60.csv", "MIN10.csv", "MIN20.csv", "MIN30.csv",
            "MIN5.csv", "MIN60.csv", "OPEN0.csv", "QTLD10.csv", "QTLD20.csv",
            "QTLD30.csv", "QTLD5.csv", "QTLD60.csv", "QTLU10.csv",
            "QTLU20.csv", "QTLU30.csv", "QTLU5.csv", "QTLU60.csv",
            "RANK10.csv", "RANK20.csv", "RANK30.csv", "RANK5.csv",
            "RANK60.csv", "RESI10.csv", "RESI20.csv", "RESI30.csv",
            "RESI5.csv", "RESI60.csv", "ROC10.csv", "ROC20.csv", "ROC30.csv",
            "ROC5.csv", "ROC60.csv"
        ]

    @property
    def processed_file_names(self):
        return [
            "timeseries_stocks_v1.pt", "spatial_graph_v1.pt",
            "temporal_graph_v1.pt"
        ]

    def download(self):
        print("Start download from HuggingFace...")
        dataset_name = "AI4FinTech/stockticker"
        self.downloaded_files = []
        for file in tqdm(self.raw_file_names):
            a_downloaded_files = hf_hub_download(
                repo_id=dataset_name,
                filename=file,
                repo_type="dataset",
            )
            self.downloaded_files.append(a_downloaded_files)

    def process(self) -> None:
        return super().process()


class StockTickerDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

    def setup(self, stage=None):
        raise NotImplementedError

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()
