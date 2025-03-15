from aggregation import aggregation_factory

from .bert import BERTTrainer
from .bert_grs import BERTGRSTrainer
from .dae import DAETrainer
from .vae import VAETrainer
from .vae_grs import VAEGRSTrainer

TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    DAETrainer.code(): DAETrainer,
    VAETrainer.code(): VAETrainer,
    BERTGRSTrainer.code(): BERTGRSTrainer,
    VAEGRSTrainer.code(): VAEGRSTrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    if args.trainer_code == BERTGRSTrainer.code() or args.trainer_code == VAEGRSTrainer.code():
        aggr_strategy = aggregation_factory(args.aggregation_code)
        return trainer(args, model, train_loader, val_loader, test_loader, export_root, aggr_strategy)
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
