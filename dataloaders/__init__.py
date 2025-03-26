import grouping
from grouping import GROUPING_STRATEGIES

from datasets import dataset_factory
from dataset import dataset_factory as rs_dataset_factory
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
        if args.mode == 'test_best':
            model_name = 'BERT4Rec' if args.dataloader_code == BertGRSDataloader.code() else 'VAE'
            train_ds, _, _ = rs_dataset_factory(args.dataset_code)
            group_strategy = GROUPING_STRATEGIES[args.grouping_code].load_with_best_hyperparams(train_ds, model_name)
        else:
            group_strategy = grouping.grouping_factory(dataset_name=args.dataset_code, group_size=args.group_size,
                                                       grouping_method=args.grouping_code, n_clusters=args.n_clusters,
                                                       similarity_threshold=args.similarity_threshold)
        dataloader = dataloader(args, dataset, group_strategy)
    else:
        dataloader = dataloader(args, dataset)

    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
