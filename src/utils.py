import os
import random
import numpy as np
import torch

from tlidb.TLiDB.datasets.get_dataset import get_dataset
from tlidb.TLiDB.data_loaders.data_loaders import get_dataloader
from tlidb.TLiDB.metrics.initializer import get_metric_computer


def load_datasets_split(split, tasks, datasets, config):
    split_datasets = {"datasets": [], "loaders": [], "metrics": []}
    for t, d in zip(tasks, datasets):
        cur_dataset = get_dataset(dataset=d, task=t, dataset_folder=config.data_dir,
                                  model_type=config.model_type,
                                  max_dialogue_length=config.max_dialogue_length,
                                  split=split, few_shot_percent=config.few_shot_percent)
        if config.frac < 1.0:
            cur_dataset.random_subsample(config.frac)

        split_datasets["datasets"].append(cur_dataset)
        split_datasets["loaders"].append(
            get_dataloader(split, cur_dataset, config.gpu_batch_size, config, collate_fn=cur_dataset.collate,
                           num_workers=config.num_workers))
        split_datasets["metrics"].append(get_metric_computer(cur_dataset.metrics, **cur_dataset.metric_kwargs))
    return split_datasets


def concat_t_d(task, dataset_name):
    return f"{task}_{dataset_name}"


def move_to(obj, device):
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int) or isinstance(obj, str) or obj is None:
        return obj
    else:
        # Assume obj is a Tensor or other type
        return obj.to(device)


def detach_and_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int) or isinstance(obj, str) or obj is None:
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")


def collate_list(inputs):
    """
    If inputs is a list of Tensors, it concatenates them all along the first dimension.

    If inputs is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If inputs is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(inputs, list):
        raise TypeError("collate_list must take in a list")
    elem = inputs[0]
    if torch.is_tensor(elem):
        return torch.cat(inputs)
    elif isinstance(elem, list):
        return [obj for sublist in inputs for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in inputs]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")


def set_seed(seed):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def save_algorithm(algorithm, epoch, best_val_metric, path):
    state = dict()
    state['algorithm'] = algorithm.state_dict()
    state['epoch'] = epoch
    state['best_val_metric'] = best_val_metric
    torch.save(state, path)


def load_algorithm(algorithm, path):
    state = torch.load(path)
    algorithm.load_state_dict(state['algorithm'])
    return state['epoch'], state['best_val_metric']


def save_pred(y_pred, path_prefix, instance_ids):
    # Single tensor
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    if torch.is_tensor(y_pred):
        y_pred_np = y_pred.numpy()
        if instance_ids is not None:
            assert (len(instance_ids) == len(y_pred_np)), "Mismatched lengths between instance_ids and y_pred"
            with open(path_prefix + '.csv', 'w') as f:
                for id, pred in zip(instance_ids, y_pred_np):
                    if isinstance(pred, np.ndarray):
                        f.write(f"{id},[{' '.join(map(str, pred))}]\n")
                    else:
                        f.write(f"{id},{pred}\n")
        else:
            np.save(path_prefix + '.csv', y_pred_np)
    # Dictionary
    elif isinstance(y_pred, dict):
        if instance_ids is not None:
            assert (len(instance_ids) == len(y_pred)), "Mismatched lengths between instance_ids and y_pred"
            with open(path_prefix + '.csv', 'w') as f:
                for k, v in y_pred.items():
                    f.write(f"{k}, \n")
                    for id, pred in zip(instance_ids, v):
                        if isinstance(pred, str):
                            pred = pred.replace(",", " ")
                        f.write(f"{id},{pred}\n")
        else:
            torch.save(y_pred, path_prefix + '.pt')
    elif isinstance(y_pred, list):
        if instance_ids is not None:
            assert (len(instance_ids) == len(y_pred)), "Mismatched lengths between instance_ids and y_pred"
            with open(path_prefix + '.csv', 'w') as f:
                for id, pred in zip(instance_ids, y_pred):
                    if isinstance(pred, str):
                        pred = pred.replace(",", " ")
                    elif isinstance(pred, list):
                        pred = "[SEPERATOR]".join(pred)
                        pred = pred.replace(",", " ")
                    f.write(f"{id},{pred}\n")
        else:
            torch.save(y_pred, path_prefix + '.pt')
    else:
        raise TypeError("Invalid type for save_pred")
