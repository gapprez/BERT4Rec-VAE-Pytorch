from aggregation import Average, AGGREGATION_METHODS
from dataset import dataset_factory
from grouping import GROUPING_STRATEGIES, FCMWithPCCGrouping
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
    for aggregation_code in tqdm([Average.code()], desc='Testing all configurations'):
        args.do_aggregation = aggregation_code is not None
        args.aggregation_code = aggregation_code if args.do_aggregation else Average.code()

        for grouping in [FCMWithPCCGrouping]:
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

def get_model_name():
    if args.dataloader_code == 'bert_grs':
        return 'BERT4Rec'
    elif args.dataloader_code == 'ae_grs':
        return 'VAE'

    raise ValueError("Cannot infer model name")

def test_best():
    for aggregation_code in tqdm([None] + list(AGGREGATION_METHODS.keys()), desc='Testing best configurations'):
        args.do_aggregation = aggregation_code is not None
        args.aggregation_code = aggregation_code if args.do_aggregation else Average.code()

        for grouping in GROUPING_STRATEGIES.values():
            args.grouping_code = grouping.code()
            args.trainer_code = get_trainer_code()

            # Setting experiment folder
            train_ds, _, _ = dataset_factory(args.dataset_code)
            grouping_method = GROUPING_STRATEGIES[args.grouping_code].load_with_best_hyperparams(train_ds, get_model_name())
            args.group_size = grouping_method.group_size
            if grouping_method.is_similarity():
                args.similarity_threshold = grouping_method.similarity_threshold
            else:
                args.n_clusters = grouping_method.n_clusters

            set_experiment_description(grouping)

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
    elif args.mode == 'test_best':
        test_best()
    else:
        raise ValueError("Wrong mode")
