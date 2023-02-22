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

    if config.do_train:
        source_tasks = [config.target_tasks, config.source_tasks, config.source_tasks + config.target_tasks]

        for i, st in enumerate(source_tasks):
            config.train_tasks = st
            config.save_path_dir = config.log_and_model_dir + '/' + str(i)

            if not os.path.exists(config.save_path_dir):
                os.makedirs(config.save_path_dir)

            set_seed(config.seed)
            datasets = dict()

            # load datasets for training
            datasets['train'] = load_datasets_split("train", config.train_tasks, config.train_datasets, config)
            datasets['dev'] = load_datasets_split("dev", config.dev_tasks, config.dev_datasets, config)

            # initialize algorithm
            algorithm = initialize_algorithm(config, datasets)

            train(algorithm, datasets, config)

    if config.do_finetune:
        assert (config.target_datasets and config.target_tasks), "Must specify target datasets and tasks to finetune"
        datasets = {}

        # if fine-tuning, set fine-tune train, and fine-tune dev to the same tasks
        config.finetune_train_tasks = config.finetune_tasks
        config.finetune_train_datasets = config.finetune_datasets
        config.finetune_dev_tasks = config.finetune_tasks
        config.finetune_dev_datasets = config.finetune_datasets

        # load datasets for fine-tuning
        datasets['train'] = load_datasets_split("train", config.finetune_train_tasks, config.finetune_train_datasets,
                                                config)
        datasets['dev'] = load_datasets_split("dev", config.finetune_dev_tasks, config.finetune_dev_datasets, config)

        config.save_path_dir = config.log_and_model_dir + '/ft'
        # create new logger for fine-tuning
        if not os.path.exists(config.save_path_dir):
            os.makedirs(config.save_path_dir)

        # initialize algorithm
        algorithm = initialize_algorithm(config, datasets)

        dummy_algo = initialize_algorithm(config, datasets)
        weights = {name: [] for name, param in algorithm.model.named_parameters()}

        # perform weight averaging
        for i in range(3):
            pth = config.log_and_model_dir + '/' + str(i) + '/best_model.pt'
            load_algorithm(dummy_algo, pth)
            for name, param in dummy_algo.model.named_parameters():
                weights[name].append(param)

        for name, param in algorithm.model.named_parameters():
            w = weights[name]
            param.data = sum(w) / len(w)
        del weights, dummy_algo, w, param

        train(algorithm, datasets, config)

    if config.do_eval:
        assert (config.target_datasets and config.target_tasks), "Must specify target datasets and tasks to finetune"
        datasets = dict()
        datasets['test'] = load_datasets_split("test", config.eval_tasks, config.eval_datasets, config)

        # initialize algorithm
        algorithm = initialize_algorithm(config, datasets)

        config.save_path_dir = config.log_and_model_dir + '/0'
        epoch, best_val_metric = load_algorithm(algorithm, config.save_path_dir + '/best_model.pt')
        evaluate(algorithm, datasets, config, epoch, is_baseline=True)

        config.save_path_dir = config.log_and_model_dir + '/ft'
        epoch, best_val_metric = load_algorithm(algorithm, config.save_path_dir + '/best_model.pt')
        evaluate(algorithm, datasets, config, epoch, is_baseline=False)


if __name__ == "__main__":
    config = parse_args()
    main(config)