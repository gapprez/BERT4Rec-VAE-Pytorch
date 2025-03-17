import torch
import torch.utils.data as data_utils
from grouping.base import GroupingStrategy
from tqdm import tqdm

from . import BertDataloader
from .bert import BertEvalDataset


class BertGrsEvalDataset(BertEvalDataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples, umap, group_strategy: GroupingStrategy):
        super().__init__(u2seq, u2answer, max_len, mask_token, negative_samples)
        self.group_strategy = group_strategy
        self.umap = umap

        users_in_groups = self.group_strategy.get_unique_users()
        self.users = []
        for user in tqdm(users_in_groups, desc="Filtering users in groups for testing"):
            u = self.umap[user]
            self.users.append(u)
        self.users = sorted(self.users)

    def get_user_from_map(self, user):
        return list(self.umap.keys())[list(self.umap.values()).index(user)]


class BertGRSWithAggEvalDataset(BertGrsEvalDataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples, umap, group_strategy: GroupingStrategy):
        super().__init__(u2seq, u2answer, max_len, mask_token, negative_samples, umap, group_strategy)

    def __getitem__(self, index):
        sequences, candidates_list, labels_list = [], [], []

        # Getting candidates and labels from main user
        main_user = self.users[index]
        main_user_map = self.get_user_from_map(main_user)
        answer = self.u2answer[main_user]
        negs = self.negative_samples[main_user]
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        group_map = self.group_strategy.get_user_group(main_user_map)
        group = [self.umap[user] for user in group_map]

        for user in group:
            seq = self.u2seq[user]
            seq = seq + [self.mask_token]
            seq = seq[-self.max_len:]  # Truncate to max length
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq  # Pad sequence

            sequences.append(torch.LongTensor(seq))

        # Stack all sequences, candidates, and labels along the first dimension
        return torch.stack(sequences), torch.LongTensor(candidates), torch.LongTensor(labels)


class BertGRSDataloader(BertDataloader):
    def __init__(self, args, dataset, group_strategy: GroupingStrategy):
        super().__init__(args, dataset)
        self.group_strategy = group_strategy
        self.do_aggregation = args.do_aggregation

    @classmethod
    def code(cls):
        return 'bert_grs'

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test

        if self.do_aggregation:
            dataset = BertGRSWithAggEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN,
                                                self.test_negative_samples, self.umap, self.group_strategy)
        else:
            dataset = BertGrsEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN,
                                         self.test_negative_samples, self.umap, self.group_strategy)

        return dataset

    @staticmethod
    def custom_collate_fn(batch):
        return batch

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        if self.do_aggregation:
            dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=False, pin_memory=True, collate_fn=self.custom_collate_fn)
        else:
            dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return dataloader
