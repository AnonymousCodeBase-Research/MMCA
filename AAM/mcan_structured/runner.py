import sys
import time
from collections import OrderedDict
import pyprind
import torch
import torch.nn as nn
import pandas as pd  # for predict()/joining attributes

from .data import MatchingIterator
from .optim import Optimizer, SoftNLLLoss
from .utils import tally_parameters


class Statistics(object):
    def __init__(self):
        self.loss_sum = 0.0
        self.examples = 0.0
        self.tps = 0.0
        self.tns = 0.0
        self.fps = 0.0
        self.fns = 0.0
        self.start_time = time.time()

    def update(self, loss=0.0, tps=0.0, tns=0.0, fps=0.0, fns=0.0):
        loss = float(loss); tps = float(tps); tns = float(tns); fps = float(fps); fns = float(fns)
        examples = tps + tns + fps + fns
        self.loss_sum += loss * examples
        self.tps += tps; self.tns += tns; self.fps += fps; self.fns += fns
        self.examples += examples

    def loss(self):
        return self.loss_sum / max(self.examples, 1.0)

    def precision(self):
        return 100.0 * self.tps / max(self.tps + self.fps, 1.0)

    def recall(self):
        return 100.0 * self.tps / max(self.tps + self.fns, 1.0)

    def f1(self):
        p = self.precision(); r = self.recall()
        return 2.0 * p * r / max(p + r, 1.0)

    def accuracy(self):
        return 100.0 * (self.tps + self.tns) / max(self.examples, 1.0)

    def examples_per_sec(self):
        elapsed = max(time.time() - self.start_time, 1e-8)
        return self.examples / elapsed


class Runner(object):
    # ---------- public helper: predict (MCA-style) ----------
    @staticmethod
    def predict(model, dataset, output_attributes=False, **kwargs):
        """
        Return a pandas.DataFrame with:
          index = id (from model.meta.id_field)
          column 'match_score' = probability for class 1 in [0,1].

        If output_attributes=True, join the raw CSV attributes onto the output.
        """
        id_col = getattr(model.meta, 'id_field', 'id')
        kwargs.setdefault('verbose', False)

        predictions = Runner._run(
            path='',
            dataset_index=0,
            run_type='PREDICT',
            model=model,
            dataset=dataset,
            train=False,
            return_predictions=True,
            **kwargs
        )

        pred_table = pd.DataFrame(predictions, columns=(id_col, 'match_score')).set_index(id_col)
        if output_attributes:
            raw_table = pd.read_csv(dataset.path).set_index(id_col)
            raw_table.index = raw_table.index.astype('str')
            pred_table = pred_table.join(raw_table)
        return pred_table

    # ---------- printing ----------
    @staticmethod
    def _print_final_stats(epoch, runtime, datatime, stats, test, verbose=True):
        if not verbose:
            return
        print((
            'Finished Epoch {epoch} || Run Time: {runtime:6.1f} | '
            'Load Time: {datatime:6.1f} || '
            'Prec: {prec:6.2f} | Rec: {rec:6.2f} | F1: {f1:6.2f} || Ex/s: {eps:6.2f}\n'
        ).format(
            epoch=epoch,
            runtime=runtime,
            datatime=datatime,
            f1=stats.f1(),
            prec=stats.precision(),
            rec=stats.recall(),
            eps=stats.examples_per_sec()
        ))

    # ---------- scores ----------
    @staticmethod
    def _compute_scores(output, target):
        predictions = torch.argmax(output, dim=1)
        correct = (predictions == target.data).float()
        incorrect = (1.0 - correct).float()
        positives = (target.data == 1).float()
        negatives = (target.data == 0).float()

        tp = torch.dot(correct, positives).item()
        tn = torch.dot(correct, negatives).item()
        fp = torch.dot(incorrect, negatives).item()
        fn = torch.dot(incorrect, positives).item()
        return tp, tn, fp, fn

    # ---------- core loop (MCA-style signature) ----------
    @staticmethod
    def _run(path,
             dataset_index,
             run_type,
             model,
             dataset,
             criterion=None,
             optimizer=None,
             train=False,
             device=None,
             batch_size=16,
             batch_callback=None,
             epoch_callback=None,
             progress_style='bar',
             log_freq=5,
             sort_in_buckets=None,
             return_predictions=False,
             verbose=True,
             **kwargs):

        def _p(*args, **kws):
            if verbose:
                print(*args, **kws)

        # device
        if device == 'cpu':
            model = model.cpu()
            if criterion:
                criterion = criterion.cpu()
        elif torch.cuda.is_available():
            model = model.cuda()
            device = 'cuda'
            if criterion:
                criterion = criterion.cuda()
        elif device == 'gpu':
            raise ValueError('No GPU available.')

        sort_in_buckets = train
        run_iter = MatchingIterator(
            dataset,
            model.meta,
            train,
            batch_size=batch_size,
            device=device,
            sort_in_buckets=sort_in_buckets
        )

        model.train(mode=train)

        if not hasattr(model, 'epoch') or model.epoch is None:
            model.epoch = 0

        epoch = model.epoch
        datatime = 0.0
        runtime = 0.0
        cum_stats = Statistics()
        stats = Statistics()
        predictions = []
        id_attr = model.meta.id_field
        label_attr = model.meta.label_field

        if train and epoch == 0:
            _p('* Number of trainable parameters:', tally_parameters(model))

        _p('===> ', run_type, f'Epoch {epoch + 1}')

        batch_end = time.time()
        # progress bar
        if verbose:
            total_ticks = max(len(run_iter) // max(int(log_freq), 1), 1)
            pbar = pyprind.ProgBar(total_ticks, bar_char='*', width=30)
            tick_every = max(int(log_freq), 1)
            tick_count = 0
            _p("0% [*] 100% | ETA: 00:00:00")
        else:
            total_ticks = 0
            tick_every = 10**9
            tick_count = 0
            pbar = None

        for batch_idx, batch in enumerate(run_iter):
            batch_start = time.time()
            datatime += batch_start - batch_end

            if train:
                # AAM: model returns output only
                output = model(batch)

                loss = float('NaN')
                if criterion:
                    loss = criterion(output, getattr(batch, label_attr))

                if hasattr(batch, label_attr):
                    scores = Runner._compute_scores(output, getattr(batch, label_attr))
                else:
                    scores = (0.0, 0.0, 0.0, 0.0)

                cum_stats.update(float(loss), *scores)
                stats.update(float(loss), *scores)

                if return_predictions:
                    # expect batch to expose ids under model.meta.id_field
                    for idx, ex_id in enumerate(getattr(batch, id_attr)):
                        predictions.append((ex_id, float(output[idx, 1].exp())))

                if (batch_idx + 1) % tick_every == 0 and verbose:
                    if tick_count < total_ticks:
                        pbar.update()
                        tick_count += 1
                    stats = Statistics()

                # backward
                model.zero_grad()
                loss.backward()
                if not optimizer.params:
                    optimizer.set_parameters(model.named_parameters())
                optimizer.step()

            else:
                with torch.no_grad():
                    output = model(batch)

                    loss = float('NaN')
                    if criterion:
                        loss = criterion(output, getattr(batch, label_attr))

                    if hasattr(batch, label_attr):
                        scores = Runner._compute_scores(output, getattr(batch, label_attr))
                    else:
                        scores = (0.0, 0.0, 0.0, 0.0)

                    cum_stats.update(float(loss), *scores)
                    stats.update(float(loss), *scores)

                    if return_predictions:
                        for idx, ex_id in enumerate(getattr(batch, id_attr)):
                            predictions.append((ex_id, float(output[idx, 1].exp())))

                    if (batch_idx + 1) % tick_every == 0 and verbose:
                        if tick_count < total_ticks:
                            pbar.update()
                            tick_count += 1
                        stats = Statistics()

            batch_end = time.time()
            runtime += batch_end - batch_start
            if verbose:
                sys.stderr.flush()

        # finish progress bar if under-updated
        if verbose:
            while tick_count < total_ticks:
                pbar.update()
                tick_count += 1

        # optional epoch callback (DeepMatcher parity)
        if epoch_callback is not None and run_type == 'TRAIN':
            epoch_info = {
                'epoch': epoch + 1,
                'precision': cum_stats.precision(),
                'recall': cum_stats.recall(),
                'f1': cum_stats.f1(),
                'accuracy': cum_stats.accuracy(),
                'loss': cum_stats.loss(),
            }
            epoch_callback(epoch_info)

        Runner._print_final_stats(epoch + 1, runtime, datatime, cum_stats,
                                  test=(run_type == 'TEST'), verbose=verbose)

        if return_predictions:
            return predictions
        else:
            # return F1 (percent scale) to mirror MCA train loop
            return cum_stats.f1()

    # ---------- train/eval (MCA-style) ----------
    @staticmethod
    def train(model,
              path,
              dataset_index,
              train_dataset,
              validation_dataset,
              best_save_path,
              epochs=10,
              criterion=None,
              optimizer=None,
              pos_neg_ratio=None,
              label_smoothing=0.05,
              save_every_prefix=None,
              save_every_freq=1,
              **kwargs):
        """Train an AAM model; returns best validation F1 (percent)."""
        epoch_callback = kwargs.pop('epoch_callback', None)
        verbose = kwargs.get('verbose', True)

        model.initialize(train_dataset)

        model._register_train_buffer('optimizer_state', None)
        model._register_train_buffer('best_score', None)
        model._register_train_buffer('epoch', None)

        if criterion is None:
            if pos_neg_ratio is None:
                pos_neg_ratio = 1
            else:
                assert pos_neg_ratio > 0
            pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)
            neg_weight = 2 - pos_weight
            # AAM SoftNLLLoss expects (input, target)
            criterion = SoftNLLLoss(label_smoothing,
                                    torch.Tensor([neg_weight, pos_weight]))

        optimizer = optimizer or Optimizer()
        if model.optimizer_state is not None:
            model.optimizer.base_optimizer.load_state_dict(model.optimizer_state)

        if model.epoch is None:
            epochs_range = range(epochs)
        else:
            epochs_range = range(model.epoch + 1, epochs)

        if model.best_score is None:
            model.best_score = -1
        optimizer.last_acc = model.best_score

        for epoch in epochs_range:
            model.epoch = epoch

            # TRAIN
            Runner._run(
                path, dataset_index, 'TRAIN', model, train_dataset,
                criterion=criterion,
                optimizer=optimizer,
                train=True,
                epoch_callback=epoch_callback,
                verbose=verbose,
                **kwargs
            )

            # VALID (return F1 percent)
            score = Runner._run(
                path, dataset_index, 'EVAL', model, validation_dataset,
                train=False, verbose=verbose, **kwargs
            )

            optimizer.update_learning_rate(score, epoch + 1)
            model.optimizer_state = optimizer.base_optimizer.state_dict()

            new_best_found = False
            if score > model.best_score:
                if verbose:
                    print('* Best F1:', score)
                model.best_score = score
                new_best_found = True
                if best_save_path and new_best_found:
                    if verbose:
                        print('Saving best model...')
                    model.save_state(best_save_path)
                    if verbose:
                        print('Done.')

            if save_every_prefix is not None and (epoch + 1) % save_every_freq == 0:
                if verbose:
                    print('Saving epoch model...')
                save_path = '{prefix}_ep{epoch}.pth'.format(
                    prefix=save_every_prefix, epoch=epoch + 1)
                model.save_state(save_path)
                if verbose:
                    print('Done.')
            if verbose:
                print('---------------------\n')

        if verbose:
            print('Loading best model...')
        model.load_state(best_save_path)
        if verbose:
            print('Training done.')

        return model.best_score

    @staticmethod
    def eval(model, path, dataset_index, dataset, **kwargs):
        """Evaluate the model on a dataset; returns F1 (percent)."""
        return Runner._run(path, dataset_index, 'TEST', model, dataset, **kwargs)
