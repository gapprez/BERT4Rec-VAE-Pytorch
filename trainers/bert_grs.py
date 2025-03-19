import json
import os

import torch
from aggregation.aggregation import AggregationScheme
from tqdm import tqdm

from utils import AverageMeterSet
from . import BERTTrainer
from .utils import recalls_and_ndcgs_for_ks


class BERTGRSTrainer(BERTTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root,
                 aggregation_strategy: AggregationScheme):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.aggregation_strategy = aggregation_strategy

    @classmethod
    def code(cls):
        return 'bert_grs'

    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        scores = []  # B x C
        labels = []  # B x C
        for (seq_group, candidates_group, labels_group) in batch:
            scores_i = self.model(seq_group)
            scores_i = scores_i[:, -1, :]
            candidates_group = candidates_group[None, :]
            scores_i = scores_i.gather(1, candidates_group.expand(seq_group.shape[0], -1))
            scores.append(self.aggregation_strategy.aggregate_pytorch(scores_i))
            labels.append(labels_group)

        metrics = recalls_and_ndcgs_for_ks(torch.stack(scores), torch.stack(labels), self.metric_ks)

        return metrics

    def test(self):
        print('Test best model with test set!')

        best_model = torch.load(f'{self.args.dataset_code}_best_acc_model.pth', map_location=torch.device(self.device)).get(
            'model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, group_batch in enumerate(tqdm_dataloader):
                batch = [(seq.to(self.device), candidates.to(self.device), labels.to(self.device)) for
                         (seq, candidates, labels) in group_batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            print(average_metrics)
