import os
from algorithms.initializer import initialize_algorithm
from train import train, evaluate
from utils import load_datasets_split, load_algorithm, set_seed
from argparser import parse_args


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

    source_tasks = [config.target_tasks, config.source_tasks, config.source_tasks + config.target_tasks]

    if config.do_train:
        for i, st in enumerate(source_tasks):
            config.train_tasks = st
            config.save_path_dir = config.log_and_model_dir + '/' + str(i)

            if not os.path.exists(config.save_path_dir):
                os.makedirs(config.save_path_dir)

            set_seed(config.seed)
            datasets = dict()

            datasets['train'] = load_datasets_split("train", config.train_tasks, config.train_datasets, config)
            datasets['dev'] = load_datasets_split("dev", config.dev_tasks, config.dev_datasets, config)

            algorithm = initialize_algorithm(config, datasets)
            train(algorithm, datasets, config)

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

        for i, st in enumerate(source_tasks):
            algorithm = initialize_algorithm(config, datasets)
            pth = config.log_and_model_dir + '/' + str(i) + '/best_model.pt'
            load_algorithm(algorithm, pth)

            config.save_path_dir = config.log_and_model_dir + f'/ft-{i}'
            if not os.path.exists(config.save_path_dir):
                os.makedirs(config.save_path_dir)
            train(algorithm, datasets, config)

    if config.do_eval:
        assert (config.target_datasets and config.target_tasks), "Must specify target datasets and tasks to finetune"
        datasets = dict()
        config.finetune_train_tasks = config.finetune_tasks
        config.finetune_train_datasets = config.finetune_datasets
        config.finetune_dev_tasks = config.finetune_tasks
        config.finetune_dev_datasets = config.finetune_datasets

        datasets['train'] = load_datasets_split("train", config.finetune_train_tasks, config.finetune_train_datasets,
                                                config)
        datasets['dev'] = load_datasets_split("dev", config.finetune_dev_tasks, config.finetune_dev_datasets, config)
        algorithm = initialize_algorithm(config, datasets)

        # perform weight averaging
        dummy_algo = initialize_algorithm(config, datasets)
        weights = {name: [] for name, param in algorithm.model.named_parameters()}

        pth = config.log_and_model_dir + '/0/best_model.pt'
        load_algorithm(algorithm, pth)
        for name, param in dummy_algo.model.named_parameters():
            weights[name].append(param)
        for i in range(1, 3):
            pth = config.log_and_model_dir + f'/ft-{i}/best_model.pt'
            load_algorithm(dummy_algo, pth)
            for name, param in dummy_algo.model.named_parameters():
                weights[name].append(param)

        for name, param in algorithm.model.named_parameters():
            w = weights[name]
            param.data = sum(w) / len(w)
        del weights, dummy_algo, w, param

        config.save_path_dir = config.log_and_model_dir + '/ft'
        if not os.path.exists(config.save_path_dir):
            os.makedirs(config.save_path_dir)
        train(algorithm, datasets, config)

        datasets = dict()
        datasets['test'] = load_datasets_split("test", config.eval_tasks, config.eval_datasets, config)
        algorithm = initialize_algorithm(config, datasets)
        # baseline
        config.save_path_dir = config.log_and_model_dir + '/0'
        epoch, best_val_metric = load_algorithm(algorithm, config.save_path_dir + '/best_model.pt')
        evaluate(algorithm, datasets, config, epoch, is_baseline=True)

        config.save_path_dir = config.log_and_model_dir + '/ft'
        epoch, best_val_metric = load_algorithm(algorithm, config.save_path_dir + '/best_model.pt')
        evaluate(algorithm, datasets, config, epoch, is_baseline=False)


if __name__ == "__main__":
    config = parse_args()
    main(config)