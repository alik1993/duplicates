import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import torch as tt
from tensorboardX import SummaryWriter

import torch as tt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TorchTrain:

    def __init__(self, model, device,
                 metrics=None, multigpu=False,
                 loss_smoothing=0.9, logdir=None, 
                 checkdir=None, verbose=True,
                 model_type='cnn'
                ):
        super().__init__()

        self.model = model
        self.multigpu = multigpu

        if multigpu:
            self.model = nn.DataParallel(self.model, device_ids=[0, 1]).cuda()

        self.logdir = logdir
        self.checkdir = checkdir
        self.verbose = verbose
        self.loss_smoothing = loss_smoothing
        self.metrics = metrics
        self.device = device
        
        self.model_type = model_type
        self.history = pd.DataFrame()

        self._curr_epoch = 0
        self._sw = SummaryWriter(self.logdir) if self.logdir is not None else None

    def _train_epoch(self, iterator, optimizer, criterion):

        self.model.train()

        running_loss = 0
        n_batches = len(iterator)
        if self.verbose:
            iterator = tqdm_notebook(iterator, total=n_batches, desc='epoch %d' % (self._curr_epoch), leave=True)

        for i, batch in enumerate(iterator):
            optimizer.zero_grad()
            if self.model_type == 'cnn':
                pred = self.model(batch['text1'].to(self.device), batch['text2'].to(self.device),
                                  batch['features'].to(self.device),
    #                               batch['length1'].to(self.device), batch['length2'].to(self.device)
                                 )
            else:
                pred = self.model(batch['text1'].to(self.device), batch['text2'].to(self.device), 
                                  batch['features'].to(self.device),
                                  batch['length1'].to(self.device), batch['length2'].to(self.device)
                                 )
            loss = criterion(pred, batch['label'].to(self.device))
            if self.multigpu:
                loss = loss.mean()
            loss.backward()
            optimizer.step()

            curr_loss = loss.data.cpu().detach().item()
            running_loss = self.loss_smoothing * running_loss + (1 - self.loss_smoothing) * curr_loss

            if self.verbose:
                iterator.set_postfix(loss='%.5f' % running_loss)

            if self._sw is not None:
                self._sw.add_scalar('Train/Loss', curr_loss, self._curr_epoch * n_batches + i)

        return running_loss

    def _test_epoch(self, iterator, criterion):
        self.model.eval()
        epoch_loss = 0

        full_pred = []
        full_label = []

        n_batches = len(iterator)
        with tt.no_grad():
            for batch in iterator:
                if self.model_type == 'cnn':
                    pred = self.model(batch['text1'].to(self.device), batch['text2'].to(self.device),
                                      batch['features'].to(self.device),
        #                               batch['length1'].to(self.device), batch['length2'].to(self.device)
                                     )
                else:
                    pred = self.model(batch['text1'].to(self.device), batch['text2'].to(self.device),
                                      batch['features'].to(self.device),
                                      batch['length1'].to(self.device), batch['length2'].to(self.device)
                                     )
#                 print(pred)
#                 print(len(pred))
#                 print(pred[0].size())
                loss = criterion(pred, batch['label'].to(self.device))
                if self.multigpu:
                    loss = loss.mean()

                epoch_loss += loss.data.item()

                full_pred.append(pred.data.cpu())
                full_label.append(batch['label'].cpu())

        full_pred = tt.cat(full_pred, dim=0)
        full_label = tt.cat(full_label, dim=0)

        val_metrics = {}
        if self.metrics is not None:
            for name, func in self.metrics.items():
                val_metrics[name] = func(full_pred, full_label)

        return epoch_loss / n_batches, val_metrics
    
    def test_res(self, iterator):
        self.model.eval()

        full_pred = []

        n_batches = len(iterator)
        with tt.no_grad():
            for batch in iterator:
                if self.model_type == 'cnn':
                    pred = self.model(batch['text1'].to(self.device), batch['text2'].to(self.device),
                                      batch['features'].to(self.device),
        #                               batch['length1'].to(self.device), batch['length2'].to(self.device)
                                     )
                else:
                    pred = self.model(batch['text1'].to(self.device), batch['text2'].to(self.device),
                                      batch['features'].to(self.device),
                                      batch['length1'].to(self.device), batch['length2'].to(self.device)
                                     )
                full_pred.append(pred.data.cpu())

        full_pred = tt.cat(full_pred, dim=0)

        return full_pred

    def train(self, train_iterator, valid_iterator, criterion, optimizer, n_epochs=100,
              scheduler=None, early_stoping=0):

        prev_loss = 100500
        es_epochs = 0
        best_epoch = None

        if self.multigpu:
            criterion = nn.parallel.DataParallel(criterion, [0, 1]).cuda()

        for epoch in range(n_epochs):
            train_loss = self._train_epoch(train_iterator, optimizer, criterion)
            valid_loss, val_metrics = self._test_epoch(valid_iterator, criterion)

            if scheduler is not None:
                scheduler.step(valid_loss)

            valid_loss = valid_loss
            if self.verbose:
                if self.metrics is not None:
                    metric_str = ''.join(['\t%s: %.5f' % (k, v) for k, v in val_metrics.items()])
                else:
                    metric_str = ''

                print(('validation loss %.5f' % valid_loss) + metric_str)

            if self._sw:
                self._sw.add_scalar('Valid/Loss', valid_loss, epoch)

            if self.checkdir is not None:
                tt.save(self.model.state_dict(), self.checkdir + '/epoch_%d_val_%f' % (epoch, valid_loss))

            record = {'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss}
            if self.metrics is not None:
                record.update(val_metrics)
            self.history = self.history.append(record, ignore_index=True)

            if early_stoping > 0:
                if valid_loss > prev_loss:
                    es_epochs += 1
                else:
                    es_epochs = 0

                if es_epochs >= early_stoping:
                    best_epoch = self.history[self.history.valid_loss == self.history.valid_loss.min()].iloc[0]
                    if self.verbose:
                        print('Early stopping! best epoch: %d val %.5f' % (best_epoch['epoch'], best_epoch['valid_loss']))
                    break

                prev_loss = min(prev_loss, valid_loss)

            self._curr_epoch += 1

        if best_epoch is not None and self.checkdir is not None:
            self.model.load_state_dict(
                tt.load(self.checkdir + '/epoch_%d_val_%f' % (best_epoch['epoch'], best_epoch['valid_loss'])))
            
            
    def test(self, test_iterator, criterion):
        return self._test_epoch(test_iterator, criterion)
    
    def test_new(self, test_iterator, criterion):
        return self.test_res(test_iterator, criterion)
    def inference(self, tokens, device):
        return model(tt.from_numpy(np.array([TEXT.vocab.stoi[x] for x in tokens]).reshape(1, -1)).to(device)).item()