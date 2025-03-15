import grouping

from datasets import dataset_factory
from .ae import AEDataloader
from .ae_grs import AEGRSDataloader
from .bert import BertDataloader
from .bert_grs import BertGRSDataloader, BertGrsEvalDataset

DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader,
    BertGRSDataloader.code(): BertGRSDataloader,
    AEGRSDataloader.code(): AEGRSDataloader
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    if args.dataloader_code == BertGRSDataloader.code() or args.dataloader_code == AEGRSDataloader.code():
        # TODO: Use argsparse in msc-grs
        group_strategy = grouping.grouping_factory(dataset_name=args.dataset_code, group_size=args.group_size,
                                                   grouping_method=args.grouping_code, n_clusters=args.n_clusters,
                                                   similarity_threshold=args.similarity_threshold)
        dataloader = dataloader(args, dataset, group_strategy)
    else:
        dataloader = dataloader(args, dataset)

    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
