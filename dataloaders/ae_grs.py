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

    @classmethod
    def code(cls):
        return 'ae_grs'

    def _combine_train_val(self):
        combined = {}
        for user in self.train.keys():
            combined[user] = self.train[user] + self.val[user]

        return combined

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = AEGRSEvalDataset(self.train, self.val, item_count=self.item_count, umap=self.umap,
                                       group_strategy=self.group_strategy)
        else:
            dataset = AEGRSEvalDataset(self._combine_train_val(), self.test, item_count=self.item_count, umap=self.umap,
                                       group_strategy=self.group_strategy)

        return dataset


class AEGRSEvalDataset(data_utils.Dataset):
    def __init__(self, user2items_input, user2items, item_count, umap, group_strategy: GroupingStrategy):
        # Split each user's items to input and label s.t. the two are disjoint
        # Both are lists of np.ndarrays
        # Filtering out users that are not in groups
        self.group_strategy = group_strategy
        users_in_groups = self.group_strategy.get_unique_users()
        self.users = [user for user in user2items_input.keys() if self.get_user_from_map(user) in users_in_groups]

        self.umap = umap
        input_list, label_list = self.__transform_input_label(user2items_input, user2items)

        # Row indices for sparse matrix
        input_user_row, label_user_row = [], []
        for user, input_items in enumerate(input_list):
            for _ in range(len(input_items)):
                input_user_row.append(user)
        for user, label_items in enumerate(label_list):
            for _ in range(len(label_items)):
                label_user_row.append(user)
        input_user_row, label_user_row = np.array(input_user_row), np.array(label_user_row)

        # Column indices for sparse matrix
        input_item_col = np.hstack(input_list)
        label_item_col = np.hstack(label_list)

        # Construct sparse matrix
        sparse_input = sparse.csr_matrix((np.ones(len(input_user_row)), (input_user_row, input_item_col)),
                                         dtype='float64', shape=(len(input_list), item_count))
        sparse_label = sparse.csr_matrix((np.ones(len(label_user_row)), (label_user_row, label_item_col)),
                                         dtype='float64', shape=(len(label_list), item_count))

        # Convert to torch tensor
        self.input_data = torch.FloatTensor(sparse_input.toarray())
        self.label_data = torch.FloatTensor(sparse_label.toarray())

    def get_user_from_map(self, user):
        return list(self.umap.keys())[list(self.umap.values()).index(user)]

    def __transform_input_label(self, inputs, labels):
        input_list, label_list = [], []

        # Removing inputs and labels from users that are not in groups
        users_in_groups = self.group_strategy.get_unique_users()
        inputs = {user: items for user, items in inputs.items() if self.get_user_from_map(user) in users_in_groups}
        labels = {user: items for user, items in labels.items() if self.get_user_from_map(user) in users_in_groups}

        assert len(inputs.keys()) == len(labels.keys())
        for items_input, items_label in zip(inputs.values(), labels.values()):
            input_list.append(np.array(items_input))
            label_list.append(np.array(items_label))

        return input_list, label_list

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        main_user = self.users[index]
        main_user_map = self.get_user_from_map(main_user)
        user_group_map = self.group_strategy.get_user_group(main_user_map)
        user_group = [self.umap[user] for user in user_group_map]
        input_data = [self.input_data[user] for user in user_group if user != main_user]
        label_data = [self.label_data[main_user_map] for user in user_group if user != main_user]

        return self.input_data[main_user], self.label_data[main_user], input_data, label_data
