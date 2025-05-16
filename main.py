import os
SLURM_TMPDIR = os.environ.get('SLURM_TMPDIR', './slurm-tmpdir')

import torch
from random import shuffle
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen, RotatedGen
import agents
from utils.utils import dotdict

import wandb

from tap import Tap
import numpy as np
import os
from tqdm import tqdm
from typing import List, Optional
import time


def compute_fgt(data):
    """
    Given a TxT data matrix, compute average forgetting at T-th task
    """
    data = np.array(data)
    num_tasks = data.shape[0]
    T = num_tasks - 1
    fgt = 0.0
    task_fgt = []
    for i in range(T):
        x = np.max(data[:T, i])
        y = data[T, i]
        print(x, y)
        fgt_i = x - y
        task_fgt.append(fgt_i)
        fgt += fgt_i
    avg_fgt = fgt/ float(num_tasks - 1)
    return avg_fgt, task_fgt


def prepare_dataloaders(args):
    # Prepare dataloaders
    Dataset = dataloaders.base.__dict__[args.dataset]

    # SPLIT CUB
    if args.is_split_cub :
        print("running split -------------")
        from dataloaders.cub import CUB
        Dataset = CUB
        if args.train_aug :
            print("train aug not supported for cub")
            return
        train_dataset, val_dataset = Dataset(args.dataroot)
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                               first_split_sz=args.first_split_size,
                                                                               other_split_sz=args.other_split_size,
                                                                               rand_split=args.rand_split,
                                                                               remap_class=not args.no_class_remap)
        n_tasks = len(task_output_space.items())
    # Permuted MNIST
    elif args.n_permutation > 0:
        # TODO : CHECK subset_size
        train_dataset, val_dataset = Dataset(args.dataroot, args.train_aug, angle=0, subset_size=args.subset_size)
        print("Working with permuatations :) ")
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                                  args.n_permutation,
                                                                                  remap_class=not args.no_class_remap)
        n_tasks = args.n_permutation
    # Rotated MNIST
    elif args.n_rotate > 0 or len(args.rotations) > 0 :
        # TODO : Check subset size
        train_dataset_splits, val_dataset_splits, task_output_space = RotatedGen(Dataset=Dataset,
                                                                                 dataroot=args.dataroot,
                                                                                 train_aug=args.train_aug,
                                                                                 n_rotate=args.n_rotate,
                                                                                 rotate_step=args.rotate_step,
                                                                                 remap_class=not args.no_class_remap,
                                                                                 rotations=args.rotations,
                                                                                 subset_size=args.subset_size)
        n_tasks = len(task_output_space.items())

    # Split MNIST
    else:
        print("running split -------------")
        # TODO : Check subset size
        train_dataset, val_dataset = Dataset(args.dataroot, args.train_aug,
                                             angle=0, subset_size=args.subset_size)
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                               first_split_sz=args.first_split_size,
                                                                               other_split_sz=args.other_split_size,
                                                                               rand_split=args.rand_split,
                                                                               remap_class=not args.no_class_remap)
        n_tasks = len(task_output_space.items())

    print(f"task_output_space {task_output_space}")

    return task_output_space, n_tasks, train_dataset_splits, val_dataset_splits


def run(args, wandb_run, task_output_space, n_tasks, train_dataset_splits, val_dataset_splits):
    # Prepare the Agent (model)
    agent_config = args
    agent_config.out_dim = {'All': args.force_out_dim} if args.force_out_dim > 0 else task_output_space
    
    val_loaders = [torch.utils.data.DataLoader(val_dataset_splits[str(task_id)],
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers)
                   for task_id in range(1, n_tasks + 1)]
    
    # agent_type : regularisation / ogd_plus / agem
    # agent_name : EWC / SI / MAS
    agent_func = agents.__dict__[args.agent_type].__dict__[args.agent_name]
    agent = agent_func(agent_config,
                       val_loaders=val_loaders,
                       wandb_run=wandb_run)
    # agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config,
    #                                                                   val_loaders=val_loaders,
    #                                                                   wandb_run=wandb_run)
    
    print("✅✅✅✅✅ USING AGENT : ", agent)
    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:', task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)
    
    # Store validation accuracies for each task at each time step
    val_accs_history = []
    
    if args.offline_training:  # Multi-task learning
        train_dataset_all = torch.utils.data.ConcatDataset(train_dataset_splits.values())
        val_dataset_all = torch.utils.data.ConcatDataset(val_dataset_splits.values())
        train_loader = torch.utils.data.DataLoader(train_dataset_all,
                                                   batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_all,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        agent.learn_batch(train_loader, val_loader)

    else:  # Continual learning
        # Compute the validation scores
        val_accs = [agent.validation(loader) for loader in val_loaders]
        print(f"val_accs : {val_accs} ")
        wandb.run.summary.update({f"R_-1": val_accs})
        val_accs_history.append(val_accs)
        num_val_samples_per_task = [len(val_dataset_splits[task_name]) for task_name in task_names]
        num_train_samples_per_task = [len(train_dataset_splits[task_name]) for task_name in task_names]
        for i in tqdm(range(len(task_names)), "task"):
            task_name = task_names[i]
            print(f'====================== Task {task_name} =======================')
            train_loader = torch.utils.data.DataLoader(train_dataset_splits[task_name],
                                                       batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers)
            val_loader = torch.utils.data.DataLoader(val_dataset_splits[task_name],
                                                     batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers)

            # Train the agent
            agent.learn_batch(train_loader, val_loader)

            val_accs = [agent.validation(loader) for loader in val_loaders]
            print(f"val_accs : {val_accs} ")
            wandb.run.summary.update({f"R_{i}": val_accs})
            val_accs_history.append(val_accs)

    # Compute forgetting metrics
    avg_forgetting, task_forgetting = compute_fgt(val_accs_history)
    avg_acc = sum(val_accs_history[-1])/len(val_accs_history[-1])
    
    # Print detailed forgetting metrics
    print("\n========== Forgetting Metrics ==========")
    print(f"Average Forgetting: {avg_forgetting:.4f}")
    print("\nPer-Task Forgetting:")
    for i, fgt in enumerate(task_forgetting):
        print(f"Task {i}: {fgt:.4f}")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Number of validation samples per task: {num_val_samples_per_task}")
    print(f"Number of training samples per task: {num_train_samples_per_task}")

    
    # Log metrics to wandb
    wandb.run.summary.update({
        "avg_forgetting": avg_forgetting,
        "task_forgetting": task_forgetting,
        "avg_acc": avg_acc,
        "num_val_samples_per_task": num_val_samples_per_task,
        "num_train_samples_per_task": num_train_samples_per_task
    })

    # Save metrics to text file
    metrics_file = os.path.join(wandb.run.dir, 'forgetting_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("========== Validation Accuracy History ==========\n")
        for i, accs in enumerate(val_accs_history):
            f.write(f"Step {i}: {accs}\n")
        
        f.write("\n========== Forgetting Metrics ==========\n")
        f.write(f"Average Forgetting: {avg_forgetting:.4f}\n")
        
        f.write("\nPer-Task Forgetting:\n")
        for i, fgt in enumerate(task_forgetting):
            f.write(f"Task {i}: {fgt:.4f}\n")
        
        f.write("\n========== Accuracy Metrics ==========\n")
        f.write(f"Average Accuracy: {avg_acc:.4f}\n")

        f.write("\n========== Number of Samples ==========\n")
        f.write(f"Number of validation samples per task: {num_val_samples_per_task}\n")
        f.write(f"Number of training samples per task: {num_train_samples_per_task}\n")

    print(f"\nMetrics saved to: {metrics_file}")

    return agent


if __name__ == '__main__':
    class Config(Tap):
        # required
        ong: bool = False
        run_name: str
        group_id: str
        
        force_out_dim: int
        no_class_remap: bool
        dataset: str


        # from add_arguments → class attrs
        gpuid: List[int] = [0]
       
        optimizer: str = "SGD"
        first_split_size: int = 2
        other_split_size: int = 2
        
        train_aug: bool = False
        rand_split: bool = False
        rand_split_order: bool = False
        schedule: List[int] = [5]
        model_weights: Optional[str] = None
        eval_on_train_set: bool = False
        offline_training: bool = False
        incremental_class: bool = False

        # rest of your existing flags
        gpu: bool = True
        workers: int = 4
        start_seed: int = 0
        end_seed: int = 5
        run_seed: int = 0
        val_size: int = 256
        lr: float = 1e-3
        scheduler: bool = False
        nepoch: int = 5
        val_check_interval: int = 300
        batch_size: int = 256
        train_percent_check: float = 1.0
        ogd_start_layer: int = 0
        ogd_end_layer: float = 1e6
        memory_size: int = 100
        hidden_dim: int = 100
        pca: bool = False
        subset_size: Optional[float] = None
        agem_mem_batch_size: int = 256
        no_transfer: bool = False
        n_permutation: int = 0
        n_rotate: int = 0
        rotate_step: int = 0
        is_split: bool = False
        data_seed: int = 2
        rotations: List[int] = []
        toy: bool = False
        ogd: bool = False
        ogd_plus: bool = False
        no_random_name: bool = False
        project: str = "testing-ogd"
        wandb_dryrun: bool = False
        wandb_dir: str = SLURM_TMPDIR
        dataroot: str = os.path.join(SLURM_TMPDIR, "datasets")
        is_split_cub: bool = False
        reg_coef: float = 0.0
        agent_type: str = "ogd_plus"
        agent_name: str = "OGD"
        model_name: str = "MLP"
        model_type: str = "mlp"
        dropout: float = 0.0
        gamma: float = 1.0
        is_stable_sgd: bool = False
        momentum: float = 0.0
        weight_decay: float = 0.0
        print_freq: float = 100
        no_val: bool = False

    # TODO : check known only
    config = Config().parse_args()
    config = config.as_dict()
    config = dotdict(config)
    print(f"🤖🤖🤖🤖 config : {config}")
    if torch.cuda.device_count() == 0 :
        config.gpu = False

    if not config.no_random_name :
        config.group_id = config.group_id
        
    if config.wandb_dryrun :
        os.environ["WANDB_MODE"] = "dryrun"

    torch.manual_seed(config.data_seed)
    np.random.seed(config.data_seed)
    task_output_space, n_tasks, train_dataset_splits, val_dataset_splits = prepare_dataloaders(config)

    print("run started !")

    name = f"{config.run_name}-{config.run_seed}"
    wandb_run = wandb.init(tags=["lightning"],
                           project=config.project,
                           sync_tensorboard=False,
                           group=config.group_id,
                           config=config,
                           job_type="eval",
                           name=name,
                           reinit=True,
                           dir=config.wandb_dir)
    # wandb_run = None
    torch.manual_seed(config.run_seed)
    np.random.seed(config.run_seed)

    start = time.time()

    agent = run(args=config,
        wandb_run=wandb_run,
        task_output_space=task_output_space,
        n_tasks=n_tasks,
        train_dataset_splits=train_dataset_splits,
        val_dataset_splits=val_dataset_splits)

    end = time.time()
    elapsed = end - start
    elapsed = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
    wandb.run.summary.update({f"time": elapsed})