import json
import os

import torch
from tqdm import tqdm

from utils import AverageMeterSet
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
        scores = []
        labels = []
        for (inputs, main_user_labels, main_user_negatives) in batch:
            logits_i, _, _ = self.model(inputs)

            mask = ((main_user_negatives != 0) | (main_user_labels != 0))
            # IMPORTANT: remove items that are not in the test set
            logits_i = logits_i[:, mask]
            logits_i = self.aggregation_method.aggregate_pytorch(logits_i)

            main_user_labels = main_user_labels[mask]

            scores.append(logits_i)
            labels.append(main_user_labels)

        metrics = recalls_and_ndcgs_for_ks(torch.stack(scores), torch.stack(labels), self.metric_ks)

        # Annealing beta
        if self.finding_best_beta:
            if self.current_best_metric < metrics[self.best_metric]:
                self.current_best_metric = metrics[self.best_metric]
                self.best_beta = self.__beta

        return metrics

    def test(self):
        print('Test best model with test set!')

        best_model = torch.load(f'{self.args.dataset_code}_best_acc_model.pth', map_location=torch.device(self.device)).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [(inp.to(self.device), lab.to(self.device), neg.to(self.device)) for
                         (inp, lab, neg) in batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            print(average_metrics)
