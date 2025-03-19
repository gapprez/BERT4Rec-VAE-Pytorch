import numpy as np
import torch
import torch.utils.data as data_utils
from grouping.base import GroupingStrategy
from scipy import sparse

from .ae import AETrainDataset, AEDataloader


class AEGRSDataloader(AEDataloader):
    def __init__(self, args, dataset, group_strategy: GroupingStrategy):
        super().__init__(args, dataset)
        self.group_strategy = group_strategy
        self.do_aggregation = args.do_aggregation

    @classmethod
    def code(cls):
        return 'ae_grs'

    def _combine_train_val(self):
        combined = {}
        for user in self.train.keys():
            combined[user] = self.train[user] + self.val[user]

        return combined

    @staticmethod
    def custom_collate_fn(batch):
        return batch

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        if self.do_aggregation:
            dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                               collate_fn=AEGRSDataloader.custom_collate_fn)
        else:
            dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        return dataloader

    def _get_eval_dataset(self, mode):
        train_data = self.train if mode == 'val' else self._combine_train_val()
        val_data = self.val if mode == 'val' else self.test

        if self.do_aggregation:
            dataset = AEGRSWithAggrEvalDataset(train_data, val_data, item_count=self.item_count, umap=self.umap,
                                               group_strategy=self.group_strategy,
                                               negative_samples=self.test_negative_samples)
        else:
            dataset = AEGRSEvalDataset(train_data, val_data, item_count=self.item_count, umap=self.umap,
                                       group_strategy=self.group_strategy, negative_samples=self.test_negative_samples)

        return dataset


class AEGRSEvalDataset(data_utils.Dataset):
    def __init__(self, user2items_input, user2items, item_count, umap, group_strategy: GroupingStrategy,
                 negative_samples):
        # Split each user's items to input and label s.t. the two are disjoint
        # Both are lists of np.ndarrays
        # Filtering out users that are not in groups
        self.group_strategy = group_strategy
        users_in_groups = self.group_strategy.get_unique_users()
        self.umap = umap
        self.users = [user for user in user2items_input.keys() if self.get_user_from_map(user) in users_in_groups]

        input_list, label_list, negative_list = self.__transform_input_label(user2items_input, user2items,
                                                                             negative_samples)

        # Row indices for sparse matrix
        input_user_row, label_user_row, negative_user_row = [], [], []
        for user, input_items in enumerate(input_list):
            for _ in range(len(input_items)):
                input_user_row.append(user)
        for user, label_items in enumerate(label_list):
            for _ in range(len(label_items)):
                label_user_row.append(user)
        input_user_row, label_user_row = np.array(input_user_row), np.array(label_user_row)
        for user, negative_items in enumerate(negative_list):
            for _ in range(len(negative_items)):
                negative_user_row.append(user)

        # Column indices for sparse matrix
        input_item_col = np.hstack(input_list)
        label_item_col = np.hstack(label_list)
        negative_item_col = np.hstack(negative_list)

        # Construct sparse matrix
        sparse_input = sparse.csr_matrix((np.ones(len(input_user_row)), (input_user_row, input_item_col)),
                                         dtype='float64', shape=(len(input_list), item_count))
        sparse_label = sparse.csr_matrix((np.ones(len(label_user_row)), (label_user_row, label_item_col)),
                                         dtype='float64', shape=(len(label_list), item_count))
        sparse_negatives = sparse.csr_matrix(
            (np.ones(len(negative_user_row)), (negative_user_row, negative_item_col)),
            dtype='float64', shape=(len(negative_list), item_count))

        # Convert to torch tensor
        self.input_data = torch.FloatTensor(sparse_input.toarray())
        self.label_data = torch.FloatTensor(sparse_label.toarray())
        self.negative_data = torch.FloatTensor(sparse_negatives.toarray())

    def get_user_from_map(self, user):
        return list(self.umap.keys())[list(self.umap.values()).index(user)]

    @staticmethod
    def __transform_input_label(inputs, labels, negatives):
        input_list, label_list, negatives_list = [], [], []

        assert len(inputs.keys()) == len(labels.keys()) == len(negatives.keys())
        for items_input, items_label, items_negative in zip(inputs.values(), labels.values(), negatives.values()):
            input_list.append(np.array(items_input))
            label_list.append(np.array(items_label))
            negatives_list.append(np.array(items_negative))

        return input_list, label_list, negatives_list

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self.input_data[index], self.label_data[index], self.negative_data[index]


class AEGRSWithAggrEvalDataset(AEGRSEvalDataset):
    def __init__(self, user2items_input, user2items, item_count, umap, group_strategy: GroupingStrategy,
                 negative_samples):
        super().__init__(user2items_input, user2items, item_count, umap, group_strategy, negative_samples)

    def __getitem__(self, index):
        main_user = self.users[index]
        main_user_map = self.get_user_from_map(main_user)
        user_group_map = self.group_strategy.get_user_group(main_user_map)
        user_group = [self.umap[user] for user in user_group_map]

        return self.input_data[user_group], self.label_data[main_user], self.negative_data[main_user]
