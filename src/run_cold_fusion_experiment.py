import os
from algorithms.initializer import initialize_algorithm
from train import train, evaluate
from utils import load_datasets_split, load_algorithm, set_seed
from argparser import parse_args
import numpy as np


# perform weight averaging
def create_wa_algorithm(config, datasets, weight_files):
    algorithm = initialize_algorithm(config, datasets)
    weights = {name: [] for name, param in algorithm.model.named_parameters()}

    for pth in weight_files:
        load_algorithm(algorithm, pth)
        for name, param in algorithm.model.named_parameters():
            weights[name].append(param)

    for name, param in algorithm.model.named_parameters():
        w = weights[name]
        param.data = sum(w) / len(w)
    return algorithm


def batch(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]


def main(config):
    if config.debug:
        config.gpu_batch_size = 1
        config.effective_batch_size = 1

    config.finetune_datasets = config.target_datasets
    config.finetune_tasks = config.target_tasks
    config.eval_datasets = config.target_datasets
    config.eval_tasks = config.target_tasks

    config.train_datasets = config.source_datasets
    config.dev_datasets = config.target_datasets
    config.dev_tasks = config.target_tasks

    # source_tasks = [config.target_tasks, config.source_tasks + config.target_tasks]
    source_tasks = config.source_tasks
    print('source_tasks', source_tasks)
    set_seed(config.seed)

    task_batch_size = 3
    num_models = np.ceil(len(source_tasks) / task_batch_size)
    weight_files = [config.log_and_model_dir + f'/{i}/best_model.pt' for i in range(num_models)]

    if config.do_train:
        # baseline
        datasets = dict()
        datasets['train'] = load_datasets_split("train", config.dev_tasks, config.dev_datasets, config)
        datasets['dev'] = load_datasets_split("dev", config.dev_tasks, config.dev_datasets, config)
        config.save_path_dir = config.log_and_model_dir + '/baseline'
        os.makedirs(config.save_path_dir)
        algorithm = initialize_algorithm(config, datasets)
        train(algorithm, datasets, config)

        # CF
        n_iter = 2 if config.debug else 10
        for j in range(n_iter):
            np.random.shuffle(source_tasks)
            sources = list(batch(source_tasks, task_batch_size))
            assert num_models == len(sources)
            for i, st in enumerate(sources):
                config.train_tasks = st
                config.save_path_dir = config.log_and_model_dir + '/' + str(i)

                datasets = dict()
                datasets['train'] = load_datasets_split("train", config.train_tasks, config.train_datasets, config)
                datasets['dev'] = load_datasets_split("dev", config.dev_tasks, config.dev_datasets, config)

                if j == 0:
                    os.makedirs(config.log_and_model_dir + '/' + str(i))
                    os.makedirs(config.log_and_model_dir + '/next-' + str(i))
                    algorithm = initialize_algorithm(config, datasets)
                else:
                    algorithm = create_wa_algorithm(config, datasets, weight_files)

                config.save_path_dir = config.log_and_model_dir + '/next-' + str(i)
                train(algorithm, datasets, config)

            for i in range(num_models):
                os.rename(config.log_and_model_dir + f'/next-{i}/best_model.pt',
                          config.log_and_model_dir + f'/{i}/best_model.pt')

    if config.do_finetune:
        assert (config.target_datasets and config.target_tasks), "Must specify target datasets and tasks to finetune"
        datasets = {}

        config.finetune_train_tasks = config.finetune_tasks
        config.finetune_train_datasets = config.finetune_datasets
        config.finetune_dev_tasks = config.finetune_tasks
        config.finetune_dev_datasets = config.finetune_datasets

        datasets['train'] = load_datasets_split("train", config.finetune_train_tasks, config.finetune_train_datasets,
                                                config)
        datasets['dev'] = load_datasets_split("dev", config.finetune_dev_tasks, config.finetune_dev_datasets, config)

        algorithm = create_wa_algorithm(config, datasets, weight_files)

        config.save_path_dir = config.log_and_model_dir + '/ft'
        if not os.path.exists(config.save_path_dir):
            os.makedirs(config.save_path_dir)
        train(algorithm, datasets, config)

    if config.do_eval:
        assert (config.target_datasets and config.target_tasks), "Must specify target datasets and tasks to finetune"

        config.finetune_train_tasks = config.finetune_tasks
        config.finetune_train_datasets = config.finetune_datasets
        config.finetune_dev_tasks = config.finetune_tasks
        config.finetune_dev_datasets = config.finetune_datasets

        datasets = dict()
        datasets['test'] = load_datasets_split("test", config.eval_tasks, config.eval_datasets, config)
        algorithm = initialize_algorithm(config, datasets)
        # baseline
        config.save_path_dir = config.log_and_model_dir + '/baseline'
        epoch, best_val_metric = load_algorithm(algorithm, config.save_path_dir + '/best_model.pt')
        evaluate(algorithm, datasets, config, epoch, is_baseline=True)

        config.save_path_dir = config.log_and_model_dir + '/ft'
        epoch, best_val_metric = load_algorithm(algorithm, config.save_path_dir + '/best_model.pt')
        evaluate(algorithm, datasets, config, epoch, is_baseline=False)


if __name__ == "__main__":
    config = parse_args()
    main(config)