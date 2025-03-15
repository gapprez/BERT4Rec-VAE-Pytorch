from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils
from scipy import sparse
import numpy as np


class AEDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        # For autoencoders, we should remove users from val/test sets
        # that rated items NOT in the training set

        # extract a list of unique items from the training set
        unique_items = set()
        for items in self.train.values():
            unique_items.update(items)

        # Then, we remove users from the val/test set.
        self.val = {user : items for user, items in self.val.items() \
                        if all(item in unique_items for item in items)}
        self.test = {user : items for user, items in self.test.items() \
                        if all(item in unique_items for item in items)}

        # re-map items
        self.smap = {s : i for i, s in enumerate(unique_items)}
        remap = lambda items: [self.smap[item] for item in items]
        self.train = {user : remap(items) for user, items in self.train.items()}
        self.val   = {user : remap(items) for user, items in self.val.items()}
        self.test  = {user : remap(items) for user, items in self.test.items()}

        # some bookkeeping
        del self.user_count
        self.item_count = len(unique_items)
        args.num_items = self.item_count

    @classmethod
    def code(cls):
        return 'ae'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = AETrainDataset(self.train, item_count=self.item_count)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _combine_train_val(self):
        combined = {}
        for user in self.train.keys():
            combined[user] = self.train[user] + self.val[user]

        return combined

    def _get_eval_dataset(self, mode):
        input_data = self.train if mode == 'val' else self._combine_train_val()
        label_data = self.val if mode == 'val' else self.test
        dataset = AEEvalDataset(input_data, label_data, item_count=self.item_count)
        return dataset


class AETrainDataset(data_utils.Dataset):
    def __init__(self, user2items, item_count):
        # Row indices for sparse matrix 
        #   e.g. [0, 0, 0, 1, 1, 4, 4, 4, 4] 
        #        when user2items = {0:[1,2,3], 1:[4,5], 4:[6,7,8,9]}
        user_row = []
        for user, useritem in enumerate(user2items.values()):
            for _ in range(len(useritem)):
                user_row.append(user)
        
        # Column indices for sparse matrix
        item_col = []
        for useritem in user2items.values():
            item_col.extend(useritem)
        
        # Construct sparse matrix
        assert len(user_row) == len(item_col)
        sparse_data = sparse.csr_matrix((np.ones(len(user_row)), (user_row, item_col)), 
                                        dtype='float64', shape=(len(user2items), item_count))
        
        # Convert to torch tensor
        self.data = torch.FloatTensor(sparse_data.toarray())

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


class AEEvalDataset(data_utils.Dataset):
    def __init__(self, user2items_inputs, user2items_labels, item_count):
        # Split each user's items to input and label s.t. the two are disjoint
        # Both are lists of np.ndarrays
        input_list, label_list = self.__transform_input_label(user2items_inputs, user2items_labels)

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

    def __transform_input_label(self, inputs, labels):
        input_list, label_list = [], []
        assert len(inputs.keys()) == len(labels.keys())
        for items_input, items_label in zip(inputs.values(), labels.values()):
            input_list.append(np.array(items_input))
            label_list.append(np.array(items_label))

        return input_list, label_list

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        return self.input_data[index], self.label_data[index]

