from dataset import SteamRSDataset
from grouping import FCMWithPCCGrouping

from datasets import dataset_factory
from trainers import BERTTrainer
from .ae import AEDataloader
from .bert import BertDataloader
from .bert_grs import BertGRSDataloader, BertGrsEvalDataset

DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader,
    BertGRSDataloader.code(): BertGRSDataloader
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    if args.dataloader_code == BertGRSDataloader.code():
        rs_dataset = SteamRSDataset()
        train_ds, val_ds, test_ds = rs_dataset.train_test_split()
        group_strategy = FCMWithPCCGrouping(train_ds)
        dataloader = dataloader(args, dataset, group_strategy)
    else:
        dataloader = dataloader(args, dataset)

    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
