import numpy as np
from tqdm import tqdm
import os
import torch
from utils import detach_and_clone, collate_list, concat_t_d, save_pred, save_algorithm
from tlidb.TLiDB.data_loaders.data_loaders import TLiDB_DataLoader


def run_epoch(algorithm, datasets, config, train):
    """
    Run one epoch of training or validation.
    Args:
        algorithm: (Algorithm) the algorithm to run
        datasets: (dict) contains all information about the datasets: splits, losses, etc.
        config: (Config) the configuration
        train: (boolean) True for training, False for validation (in val mode).
    """
    if train:
        algorithm.train()
        torch.set_grad_enabled(True)
    else:
        algorithm.eval()
        torch.set_grad_enabled(False)

    # convert all datasets into a single multi-task and multi-domain dataloader
    dataloader = TLiDB_DataLoader(datasets)
    task_datasets = [concat_t_d(d.task, d.dataset_name) for d in datasets['datasets']]

    epoch_y_true = {t_d: [] for t_d in task_datasets}
    epoch_y_pred = {t_d: [] for t_d in task_datasets}

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment the step
    pbar = tqdm(dataloader) if config.progress_bar else dataloader
    # cumulative loss during the epoch
    total_loss = {t_d: 0 for t_d in task_datasets}
    loss_names = {t_d: None for t_d in task_datasets}
    step = {t_d: 0 for t_d in task_datasets}
    loss_divisor = {t_d: 0 for t_d in task_datasets}  # used for averaging loss
    for batch in pbar:
        _, _, batch_metadata = batch
        batch_t_d = concat_t_d(batch_metadata['task'], batch_metadata['dataset_name'])
        if train:
            batch_results = algorithm.update(batch, sum(step.values()))
        else:
            batch_results = algorithm.evaluate(batch)

        # These should already be detached, but in some versions they won't get garbage
        #   collected properly if not detached again
        epoch_y_true[batch_t_d].append(detach_and_clone(batch_results['y_true']))
        epoch_y_pred[batch_t_d].append(detach_and_clone(batch_results['y_pred']))
        total_loss[batch_t_d] += detach_and_clone(batch_results['objective']['loss_value'])
        loss_divisor[batch_t_d] += detach_and_clone(batch_results['batch_loss_divisor'])

        # Save the name of the loss on first encounter
        if loss_names[batch_t_d] is None:
            loss_names[batch_t_d] = batch_results['objective']['loss_name']

        desc = "Train losses" if train else "Validation losses"
        for t_d in task_datasets:
            desc += f" | {t_d}: {total_loss[t_d] / loss_divisor[batch_t_d]:0.4f}"

        pbar.set_description(desc)
        step[batch_t_d] += 1

        if config.debug:
            break

    for t_d in task_datasets:
        epoch_y_true[t_d] = collate_list(epoch_y_true[t_d])
        epoch_y_pred[t_d] = collate_list(epoch_y_pred[t_d])

    # This loop is determined by the model/task/mode(train/val)
    results = {}
    if algorithm.requires_metric_calculation():
        for m, d in zip(datasets['metrics'], datasets['datasets']):
            t_d = concat_t_d(d.task, d.dataset_name)
            result_str = f'Loss-{loss_names[t_d]}: {total_loss[t_d] / loss_divisor[t_d]:0.4f}\n'

            # during training, validate response generation on language modeling loss, not evaluation metrics
            if d.task != 'response_generation':
                r, r_str = m.compute(epoch_y_pred[t_d], epoch_y_true[t_d])
                result_str += r_str
            else:
                # use negative loss as metric so that loss closest to 0 is best
                r = {loss_names[t_d]: -total_loss[t_d] / loss_divisor[t_d]}

            results[t_d] = r
            if debug:
                break
    return results, epoch_y_pred


def train(algorithm, datasets, config, best_val_metric=None):
    save_path = os.path.join(config.save_path_dir, "best_model.pt")
    n_epochs = config.num_epochs
    if config.debug:
        n_epochs = 1
    if best_val_metric is None:
        best_val_metric = -np.inf

    for epoch in range(n_epochs):
        # train
        run_epoch(algorithm, datasets['train'], config, train=True)

        # allow for training without dev set, will not save model
        if not datasets['dev'].get('datasets', None):
            continue

        # evaluate on validation set
        val_results, y_pred = run_epoch(algorithm, datasets['dev'], config, train=False)
        val_metrics = [val_results[d][m] for d in val_results for m in val_results[d]]
        cur_val_metric = sum(val_metrics) / len(val_metrics)

        if cur_val_metric > best_val_metric:
            best_val_metric = cur_val_metric
            save_algorithm(algorithm, epoch, best_val_metric, save_path)

    return best_val_metric


def evaluate(algorithm, datasets, config, epoch):
    algorithm.eval()
    torch.set_grad_enabled(False)
    for split in datasets:
        for dataset, loader, metric in zip(datasets[split]['datasets'], datasets[split]['loaders'],
                                           datasets[split]['metrics']):
            epoch_y_true = []
            epoch_y_pred = []
            epoch_instance_ids = []

            pbar = tqdm(iter(loader)) if config.progress_bar else iter(loader)
            total_loss = 0
            loss_divisor = 0

            for batch in pbar:
                # add batch metadata to the batch
                X, y, batch_metadata = batch
                batch_metadata['task'] = dataset.task
                batch_metadata['dataset_name'] = dataset.dataset_name
                batch_metadata['task_metadata'] = dataset.task_metadata
                batch = (X, y, batch_metadata)
                epoch_instance_ids.append(batch_metadata['instance_id'])

                batch_results = algorithm.evaluate(batch)
                epoch_y_true.append(detach_and_clone(batch_results['y_true']))
                y_pred = detach_and_clone(batch_results['y_pred'])
                epoch_y_pred.append(y_pred)

                total_loss += detach_and_clone(batch_results['objective']['loss_value'])
                loss_divisor += detach_and_clone(batch_results['batch_loss_divisor'])

                desc = f"Test losses | {total_loss / loss_divisor:0.4f}"
                pbar.set_description(desc)

                if config.debug:
                    break

            epoch_y_pred = collate_list(epoch_y_pred)
            epoch_y_true = collate_list(epoch_y_true)
            epoch_instance_ids = collate_list(epoch_instance_ids)

            # further unpack if instance ids are a list of lists
            if isinstance(epoch_instance_ids[0], list):
                if isinstance(epoch_y_pred, list):
                    epoch_y_pred = collate_list(epoch_y_pred)
                    epoch_y_true = collate_list(epoch_y_true)
                elif isinstance(epoch_y_pred, torch.Tensor):
                    epoch_y_pred = torch.flatten(epoch_y_pred)
                    epoch_y_true = torch.flatten(epoch_y_true)
                epoch_instance_ids = collate_list(epoch_instance_ids)

            result_str = f"Eval on {split} split at epoch {epoch}: {dataset.dataset_name} {dataset.task}-\n"
            result_str += f"Loss-{batch_results['objective']['loss_name']}: {total_loss / loss_divisor:0.4f}\n"
            r, r_str = metric.compute(epoch_y_pred, epoch_y_true)
            r['epoch'] = epoch
            result_str += r_str

            save_pred(epoch_y_pred, os.path.join(config.save_path_dir, f"{split}-predictions"), epoch_instance_ids)
