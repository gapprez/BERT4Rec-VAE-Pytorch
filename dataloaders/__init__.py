from dataset import dataset_factory as rs_dataset_factory
from grouping import FCMWithPCCGrouping

from datasets import dataset_factory
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
        train_ds, val_ds, test_ds = rs_dataset_factory(args.dataset_code)
        group_strategy = FCMWithPCCGrouping(train_ds, group_size=args.group_size, n_clusters=args.n_clusters)
        dataloader = dataloader(args, dataset, group_strategy)
    else:
        dataloader = dataloader(args, dataset)

    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
