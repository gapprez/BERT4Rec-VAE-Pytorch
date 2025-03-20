from aggregation import Average, BordaCount
from dataset import SteamRSDataset, MINDRSDataset, MovieLensRSDataset, ML1m
from grouping import FCMWithPCCGrouping, KNNGrouping, ContentBasedPCCGrouping
from tqdm import tqdm

from dataloaders import dataloader_factory
from models import model_factory
from options import args
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()


def test():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.test()


def set_experiment_description():
    args.experiment_description = f"test_{args.dataset_code}_{args.grouping_code}"
    if is_similarity_grouping(args.grouping_code):
        args.experiment_description = f"{args.experiment_description}_{args.similarity_threshold}"
    else:
        args.experiment_description = f"{args.experiment_description}_{args.n_clusters}"

    args.experiment_description = f"{args.experiment_description}_{args.group_size}"
    if args.do_aggregation:
        args.experiment_description = f"{args.experiment_description}_{args.aggregation_code}"


def test_all_config():
    for aggregation_code in tqdm([None, Average.code(), BordaCount.code()], desc='Testing all configurations'):
        args.do_aggregation = aggregation_code is not None
        args.aggregation_code = aggregation_code if args.do_aggregation else Average.code()

        for grouping_code in [FCMWithPCCGrouping.code(), ContentBasedPCCGrouping.code(), KNNGrouping.code()]:
            args.grouping_code = grouping_code
            is_similarity = is_similarity_grouping(args.grouping_code)
            n_clusters_or_sim = get_n_clusters_similarity_list()

            for clust_sim in n_clusters_or_sim:
                args.n_clusters = clust_sim if not is_similarity else args.n_clusters
                args.similarity_threshold = clust_sim if is_similarity else args.similarity_threshold

                for group_size in get_group_size_list():
                    args.group_size = group_size
                    set_experiment_description()
                    args.trainer_code = get_trainer_code()

                    test()


def get_n_clusters_similarity_list():
    dataset_value = args.dataset_code
    grouping_value = args.grouping_code

    # Match the tuple (dataset_value, grouping_value)
    if dataset_value in [SteamRSDataset.code(), MINDRSDataset.code()] and grouping_value == FCMWithPCCGrouping.code():
        return [10, 20]

    elif dataset_value in [MovieLensRSDataset.code(), ML1m.code()] and grouping_value == FCMWithPCCGrouping.code():
        return [5, 10, 20]

    elif dataset_value in [SteamRSDataset.code(), MINDRSDataset.code()] and grouping_value == KNNGrouping.code():
        return [1.4, 1.6]

    elif grouping_value == ContentBasedPCCGrouping.code():
        return [0.8, 0.9]

    elif dataset_value in [MovieLensRSDataset.code(), ML1m.code()] and grouping_value == KNNGrouping.code():
        return [1.4, 1.6]

    else:
        raise Exception(f"Error getting n_clusters/similarity list: ({dataset_value}, {grouping_value})")


def get_group_size_list():
    # Match the tuple (dataset_value, grouping_value)
    dataset_value = args.dataset_code
    grouping_value = args.grouping_code

    if dataset_value in [SteamRSDataset.code(), MINDRSDataset.code()] and \
            grouping_value in [KNNGrouping.code(), FCMWithPCCGrouping.code()]:
        return [50, 100]

    elif dataset_value in [SteamRSDataset.code(), MINDRSDataset.code()] and \
            grouping_value == ContentBasedPCCGrouping.code():
        return [5, 10]

    elif dataset_value in [MovieLensRSDataset.code(), ML1m.code()]:
        return [5, 10]

    else:
        raise Exception(f"Error getting group_size list: ({dataset_value}, {grouping_value})")


def is_similarity_grouping(grouping_code):
    return grouping_code in [KNNGrouping.code(), ContentBasedPCCGrouping.code()]


def get_trainer_code():
    if args.dataloader_code == 'ae_grs':
        return 'vae_grs' if args.do_aggregation else 'vae'
    if args.dataloader_code == 'bert_grs':
        return 'bert_grs' if args.do_aggregation else 'bert'

    raise Exception("Wrong dataloader code")


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test_all_config()
    else:
        raise ValueError('Invalid mode')
