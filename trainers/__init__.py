from aggregation.aggregation import Average

from .bert import BERTTrainer
from .bert_grs import BERTGRSTrainer
from .dae import DAETrainer
from .vae import VAETrainer

TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    DAETrainer.code(): DAETrainer,
    VAETrainer.code(): VAETrainer,
    BERTGRSTrainer.code(): BERTGRSTrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    if args.trainer_code == BERTGRSTrainer.code():
        aggr_strategy = Average()
        return trainer(args, model, train_loader, val_loader, test_loader, export_root, aggr_strategy)
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
