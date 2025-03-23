from aggregation import Average, AGGREGATION_METHODS
from grouping import GROUPING_STRATEGIES
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


def set_experiment_description(grouping):
    args.experiment_description = f"test_{args.dataset_code}_{args.grouping_code}"
    if grouping.is_similarity():
        args.experiment_description = f"{args.experiment_description}_{args.similarity_threshold}"
    else:
        args.experiment_description = f"{args.experiment_description}_{args.n_clusters}"

    args.experiment_description = f"{args.experiment_description}_{args.group_size}"
    if args.do_aggregation:
        args.experiment_description = f"{args.experiment_description}_{args.aggregation_code}"


def test_all_config():
    for aggregation_code in tqdm([None] + list(AGGREGATION_METHODS.keys()), desc='Testing all configurations'):
        args.do_aggregation = aggregation_code is not None
        args.aggregation_code = aggregation_code if args.do_aggregation else Average.code()

        for grouping in GROUPING_STRATEGIES.values():
            args.grouping_code = grouping.code()
            is_similarity = grouping.is_similarity()
            n_clusters_or_sim = grouping.get_sim_list(args.dataset_code) if is_similarity else grouping.get_n_clusters_list(args.dataset_code)

            for clust_sim in n_clusters_or_sim:
                args.n_clusters = clust_sim if not is_similarity else args.n_clusters
                args.similarity_threshold = clust_sim if is_similarity else args.similarity_threshold

                for group_size in grouping.get_group_size_list(args.dataset_code):
                    args.group_size = group_size
                    set_experiment_description(grouping)
                    args.trainer_code = get_trainer_code()

                    test()


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
