import copy
import logging
import sys
import time
import warnings
from collections import OrderedDict

import pandas as pd

import pyprind
import torch
from tqdm import tqdm

from .data import MatchingIterator
from .optim import Optimizer, SoftNLLLoss
from .utils import tally_parameters

try:
    get_ipython  # noqa: F401
    from tqdm import tqdm_notebook as tqdm  # type: ignore
except NameError:
    from tqdm import tqdm  # type: ignore

logger = logging.getLogger(__name__)


class Statistics(object):
    def __init__(self):
        self.loss_sum = 0.0
        self.examples = 0
        self.tps = 0.0
        self.tns = 0.0
        self.fps = 0.0
        self.fns = 0.0
        self.start_time = time.time()

    def update(self, loss=0.0, tps=0.0, tns=0.0, fps=0.0, fns=0.0):
        examples = int(tps + tns + fps + fns)
        self.loss_sum += float(loss) * examples
        self.tps += float(tps)
        self.tns += float(tns)
        self.fps += float(fps)
        self.fns += float(fns)
        self.examples += examples

    def loss(self):
        return self.loss_sum / max(self.examples, 1)

    def precision(self):
        return 100.0 * self.tps / max(self.tps + self.fps, 1.0)

    def recall(self):
        return 100.0 * self.tps / max(self.tps + self.fns, 1.0)

    def f1(self):
        prec = self.precision()
        rec = self.recall()
        return 2.0 * prec * rec / max(prec + rec, 1.0)

    def accuracy(self):
        return 100.0 * (self.tps + self.tns) / max(self.examples, 1)

    def examples_per_sec(self):
        return self.examples / max(time.time() - self.start_time, 1e-8)


class Runner(object):
    @staticmethod
    def _print_stats(name, epoch, batch, n_batches, stats, cum_stats):
        print((
            ' | {name} | [{epoch}][{batch:4d}/{n_batches}] || '
            'Loss: {loss:7.4f} | F1: {f1:7.2f} | Prec: {prec:7.2f} | Rec: {rec:7.2f} || '
            'Cum. F1: {cf1:7.2f} | Cum. Prec: {cprec:7.2f} | Cum. Rec: {crec:7.2f} || '
            'Ex/s: {eps:6.1f}'
        ).format(
            name=name,
            epoch=epoch,
            batch=batch,
            n_batches=n_batches,
            loss=stats.loss(),
            f1=stats.f1(),
            prec=stats.precision(),
            rec=stats.recall(),
            cf1=cum_stats.f1(),
            cprec=cum_stats.precision(),
            crec=cum_stats.recall(),
            eps=cum_stats.examples_per_sec()
        ))

    @staticmethod
    def _print_final_stats(epoch, runtime, datatime, stats):
        print((
            'Finished Epoch {epoch} || Run Time: {runtime:6.1f} | Load Time: {datatime:6.1f} || '
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

    @staticmethod
    def _set_pbar_status(pbar, stats, cum_stats):
        postfix_dict = OrderedDict([
            ('Loss', '{0:7.4f}'.format(stats.loss())),
            ('F1', '{0:7.2f}'.format(stats.f1())),
            ('Cum. F1', '{0:7.2f}'.format(cum_stats.f1())),
            ('Ex/s', '{0:6.1f}'.format(cum_stats.examples_per_sec())),
        ])
        pbar.set_postfix(ordered_dict=postfix_dict)

    @staticmethod
    def _compute_scores(output, target):
        predictions = output.max(1)[1].data
        correct = (predictions == target.data).float()
        incorrect = (1.0 - correct).float()
        positives = (target.data == 1).float()
        negatives = (target.data == 0).float()

        tp = torch.dot(correct, positives)
        tn = torch.dot(correct, negatives)
        fp = torch.dot(incorrect, negatives)
        fn = torch.dot(incorrect, positives)

        return tp, tn, fp, fn

    @staticmethod
    def _run(run_type,
             model,
             dataset,
             criterion=None,
             optimizer=None,
             train=False,
             device=None,
             batch_size=32,
             batch_callback=None,
             epoch_callback=None,
             progress_style='bar',
             log_freq=5,
             sort_in_buckets=None,
             return_predictions=False,
             **kwargs):

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'gpu':
            device = 'cuda'

        sort_in_buckets = train
        run_iter = MatchingIterator(
            dataset,
            model.meta,
            train,
            batch_size=batch_size,
            device=device,
            sort_in_buckets=sort_in_buckets)

        model = model.to(device)
        if criterion:
            criterion = criterion.to(device)

        model.train(mode=train)

        epoch = model.epoch
        show_logs = (run_type in ('TRAIN'))  # only TRAIN prints; EVAL/PREDICT are silent here
        datatime = 0.0
        runtime = 0.0
        cum_stats = Statistics()
        stats = Statistics()
        predictions = []
        id_attr = model.meta.id_field
        label_attr = model.meta.label_field

        if train and epoch == 0:
            print('* Number of trainable parameters:', tally_parameters(model))

        epoch_str = 'Epoch {0:d}'.format(epoch + 1)
        if show_logs:
            print('===> ', run_type, epoch_str)
        batch_end = time.time()

        if show_logs:
            total_ticks = max(len(run_iter) // max(int(log_freq), 1), 1)
            if progress_style == 'tqdm-bar':
                pbar = tqdm(total=total_ticks, bar_format='{l_bar}{bar}{postfix}', file=sys.stdout)
            elif progress_style == 'bar':
                pbar = pyprind.ProgBar(total_ticks, bar_char='█', width=30)
            else:
                pbar = None

        next_pred_id = 0

        for batch_idx, batch in enumerate(run_iter):
            batch_start = time.time()
            datatime += batch_start - batch_end

            output = model(batch)

            loss = float('NaN')
            if criterion:
                loss = criterion(output, getattr(batch, label_attr))

            if hasattr(batch, label_attr):
                scores = Runner._compute_scores(output, getattr(batch, label_attr))
            else:
                scores = [0] * 4

            cum_stats.update(float(loss), *scores)
            stats.update(float(loss), *scores)

            if return_predictions:
                ids_iter = getattr(batch, id_attr, None)
                if ids_iter is None:
                    ids_iter = [str(next_pred_id + i) for i in range(output.size(0))]
                    next_pred_id += output.size(0)

                for idx, id_val in enumerate(ids_iter):
                    predictions.append((id_val, float(output[idx, 1].exp())))

            if (batch_idx + 1) % max(int(log_freq), 1) == 0:
                if show_logs:
                    if progress_style == 'log':
                        Runner._print_stats(run_type, epoch + 1, batch_idx + 1, len(run_iter),
                                            stats, cum_stats)
                    elif progress_style == 'tqdm-bar' and pbar is not None:
                        pbar.update()
                        Runner._set_pbar_status(pbar, stats, cum_stats)
                    elif progress_style == 'bar' and pbar is not None:
                        pbar.update()
                stats = Statistics()

            if train:
                model.zero_grad()
                loss.backward()  # type: ignore
                if not optimizer.params:
                    optimizer.set_parameters(model.named_parameters())
                optimizer.step()

            batch_end = time.time()
            runtime += batch_end - batch_start

        if show_logs:
            if progress_style == 'tqdm-bar' and pbar is not None:
                pbar.close()
            elif progress_style == 'bar':
                sys.stderr.flush()

        # Build epoch info for TRAIN; script decides what to do next (EVAL/threshold/etc.)
        epoch_info = None
        if run_type == 'TRAIN':
            epoch_info = {
                'epoch': epoch + 1,
                'precision': cum_stats.precision(),
                'recall': cum_stats.recall(),
                'f1': cum_stats.f1(),
                'accuracy': cum_stats.accuracy(),
                'loss': cum_stats.loss(),
            }

        if show_logs:
            Runner._print_final_stats(epoch + 1, runtime, datatime, cum_stats)

        if epoch_callback is not None and run_type == 'TRAIN':
            epoch_callback(epoch_info)

        if return_predictions:
            return predictions
        else:
            return cum_stats.f1()

    @staticmethod
    def train(model,
              train_dataset,
              validation_dataset,
              best_save_path,
              epochs=30,
              criterion=None,
              optimizer=None,
              pos_neg_ratio=None,
              pos_weight=None,
              label_smoothing=0.05,
              save_every_prefix=None,
              save_every_freq=1,
              **kwargs):
        """Adds `internal_eval` kwarg:
           - internal_eval=True (default): original behavior (does DM EVAL/best/early stop)
           - internal_eval=False: skip DM EVAL, best printing, DM early stop, and skip final load
        """
        epoch_callback = kwargs.pop('epoch_callback', None)
        early_stop_patience = kwargs.pop('early_stop_patience', None)
        _ = kwargs.pop('early_stop_metric', 'f1')
        internal_eval = kwargs.pop('internal_eval', True)

        # Normalize patience: <=0 disables DM early stop
        if isinstance(early_stop_patience, (int, float)) and early_stop_patience <= 0:
            early_stop_patience = None

        model.initialize(train_dataset)

        model._register_train_buffer('optimizer_state', None)
        model._register_train_buffer('best_score', None)
        model._register_train_buffer('epoch', None)

        if criterion is None:
            if pos_weight is not None:
                assert pos_weight < 2
                warnings.warn('"pos_weight" is deprecated; use "pos_neg_ratio" instead',
                              DeprecationWarning)
                assert pos_neg_ratio is None
            else:
                if pos_neg_ratio is None:
                    pos_neg_ratio = 1
                else:
                    assert pos_neg_ratio > 0
                pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)
            neg_weight = 2 - pos_weight
            criterion = SoftNLLLoss(label_smoothing, torch.Tensor([neg_weight, pos_weight]))

        optimizer = optimizer or Optimizer()
        if model.optimizer_state is not None:
            model.optimizer.base_optimizer.load_state_dict(model.optimizer_state)

        epochs_range = range(epochs) if model.epoch is None else range(model.epoch + 1, epochs)

        if model.best_score is None:
            model.best_score = -1
        optimizer.last_acc = model.best_score

        no_improve_count = 0  # (unused if internal_eval=False)

        for epoch in epochs_range:
            model.epoch = epoch

            # TRAIN pass
            Runner._run(
                'TRAIN', model, train_dataset,
                criterion=criterion,
                optimizer=optimizer,
                train=True,
                epoch_callback=epoch_callback,
                **kwargs
            )

            if internal_eval:
                # DM's internal EVAL & early-stop logic (unchanged)
                score = Runner._run('EVAL', model, validation_dataset, train=False, **kwargs)

                optimizer.update_learning_rate(score, epoch + 1)
                model.optimizer_state = optimizer.base_optimizer.state_dict()

                new_best_found = False
                if score > model.best_score:
                    print('* Best F1:', score)
                    model.best_score = score
                    new_best_found = True
                    no_improve_count = 0
                    if best_save_path:
                        print('Saving best model...')
                        model.save_state(best_save_path)
                        print('Done.')
                else:
                    no_improve_count += 1
                    if early_stop_patience is not None and no_improve_count >= early_stop_patience:
                        print(f'Early stopping at epoch {epoch + 1}: '
                              f'no improvement in {early_stop_patience} consecutive epochs.')
                        break

                if epoch_callback is not None and new_best_found:
                    try:
                        epoch_callback({'epoch': epoch + 1, 'is_best_epoch': True})
                    except TypeError:
                        epoch_callback({'epoch': epoch + 1, 'is_best_epoch': True})

            else:
                # No DM internal eval/best/early-stop
                model.optimizer_state = optimizer.base_optimizer.state_dict()

            # Optional per-epoch snapshots (we rarely use these now)
            if save_every_prefix is not None and (epoch + 1) % save_every_freq == 0:
                print('Saving epoch model...')
                save_path = f'{save_every_prefix}_ep{epoch + 1}.pth'
                model.save_state(save_path)
                print('Done.')
            print('---------------------\n')

        # Only load best if DM actually managed it
        if best_save_path:
            print('Loading best model...')
            model.load_state(best_save_path)
            print('Training done.')

        return model.best_score

    def eval(model, dataset, **kwargs):
        return Runner._run('EVAL', model, dataset, **kwargs)

    def predict(model, dataset, output_attributes=False, **kwargs):
        model = copy.deepcopy(model)
        model._reset_embeddings(dataset.vocabs)
        id_col = getattr(model.meta, 'id_field', 'id')
        predictions = Runner._run('PREDICT', model, dataset, return_predictions=True, **kwargs)
        pred_table = pd.DataFrame(predictions, columns=(id_col, 'match_score')).set_index(id_col)
        if output_attributes:
            raw_table = pd.read_csv(dataset.path).set_index(id_col)
            raw_table.index = raw_table.index.astype('str')
            pred_table = pred_table.join(raw_table)
        return pred_table
