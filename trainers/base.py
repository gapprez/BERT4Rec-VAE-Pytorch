import json
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils
from torch.utils.tensorboard import SummaryWriter
from torch_xla import dist

from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from loggers import *
from utils import AverageMeterSet
from torch_xla import runtime as xr


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args = args

        self.model = model
        self.device = args.device
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            print("Using more than one GPU")
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.export_root = export_root

        self.logger_service = None
        self.log_period_as_iter = args.log_period_as_iter

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @abstractmethod
    def log_extra_val_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    def __maybe_add_train_sampler(self):
        dataset = self.train_loader.dataset
        if xr.world_size() > 1:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=xr.world_size(),
                rank=xr.global_ordinal(),
                shuffle=True)
            data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                     shuffle=False if self.train_sampler else True, num_workers=self.args.num_workers,
                                     drop_last=self.args.drop_last)
        else:
            return self.train_loader

    def train(self):
        dist.init_process_group('xla', init_method='xla://')

        self.train_loader = self.__maybe_add_train_sampler()

        self.device = xla.device()
        xm.master_print("Before saving model to device")
        self.model.to(self.device)

        # Initialization is nondeterministic with multiple threads in PjRt.
        # Synchronize model parameters across replicas manually.
        xm.broadcast_master_param(self.model)

        xm.master_print("Saved model to device")
        # Create loggers for master process
        if xm.is_master_ordinal():
            self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
            self.logger_service = LoggerService(self.train_loggers, self.val_loggers)
            self.add_extra_loggers()

        accum_iter = 0
        xm.master_print("Before first validation")
        self.validate(0, accum_iter)
        xm.master_print("After first validation")

        for epoch in range(self.num_epochs):
            xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
            self.validate(epoch, accum_iter)
            xm.master_print('Epoch {} test end {}'.format(epoch, test_utils.now()))
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()

        average_meter_set = AverageMeterSet()
        mp_dataloader = pl.MpDeviceLoader(self.train_loader, self.device)

        for batch_idx, batch in enumerate(mp_dataloader):
            batch_size = batch[0].size(0)

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()

            xm.optimizer_step(self.optimizer)

            if self.args.enable_lr_schedule and xm.is_master_ordinal():
                self.lr_scheduler.step()

            average_meter_set.update('loss', loss.item())
            xm.master_print('Epoch {}, loss {:.3f} '.format(epoch + 1, average_meter_set['loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch + 1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            mp_dataloader = pl.MpDeviceLoader(self.val_loader, self.device)
            for batch_idx, batch in enumerate(mp_dataloader):
                average_meter_set = self._recompute_metrics(batch, average_meter_set)

            if xm.is_master_ordinal():
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch + 1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_val_info(log_data)
                self.logger_service.log_val(log_data)

    def _recompute_metrics(self, batch, average_meter_set):
        metrics = self.calculate_metrics(batch)

        for k, v in metrics.items():
            average_meter_set.update(k, xm.mesh_reduce(k, v, np.mean))

        description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
                              ['Recall@%d' % k for k in self.metric_ks[:3]]
        description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        description = description.format(*(average_meter_set[k].avg for k in description_metrics))
        xm.master_print(description)

        return average_meter_set

    def test(self):
        print('Test best model with test set!')

        best_model = torch.load(f'{self.args.dataset_code}_best_acc_model.pth',
                                map_location=torch.device(self.device)).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            mp_loader = pl.MpDeviceLoader(self.test_loader, self.device)
            for batch_idx, batch in enumerate(mp_loader):
                average_meter_set = self._recompute_metrics(batch, average_meter_set)

            if xm.is_master_ordinal():
                average_metrics = average_meter_set.averages()
                with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                    json.dump(average_metrics, f, indent=4)

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                             momentum=args.momentum)
        else:
            raise ValueError

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
