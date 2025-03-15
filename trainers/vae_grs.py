import torch

from . import VAETrainer
from .utils import recalls_and_ndcgs_for_ks


class VAEGRSTrainer(VAETrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, aggregation_method):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.aggregation_method = aggregation_method

    @classmethod
    def code(cls):
        return 'vae_grs'

    def calculate_metrics(self, batch):
        main_user_input, main_user_label, inputs, labels = batch
        total_logits = torch.zeros((len(inputs)+1, main_user_input.shape[0]))
        total_logits[0, :] = self.model(main_user_input)

        for i, (input, label) in enumerate(zip(inputs, labels), start=1):
            logits, _, _ = self.model(input)
            total_logits[i, :] = logits

        logits = self.aggregation_method.aggregate_pytorch(total_logits)
        logits[main_user_input != 0] = -float("Inf")  # IMPORTANT: remove items that were in the input
        metrics = recalls_and_ndcgs_for_ks(logits, main_user_label, self.metric_ks)

        # Annealing beta
        if self.finding_best_beta:
            if self.current_best_metric < metrics[self.best_metric]:
                self.current_best_metric = metrics[self.best_metric]
                self.best_beta = self.__beta

        return metrics
