import torch

from datasets import dataset_factory
from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    dataset = dataset_factory(args)
    model_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}.pth' \
        .format(dataset.code(), dataset.min_rating, dataset.min_uc, dataset.min_sc, dataset.split)
    model.save_model(model_name)

    test_model = True # (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
