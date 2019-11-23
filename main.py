import argparse
import os

import torch
import torch.multiprocessing as mp

from model import ActorCritic
from train import train
import my_optim
from scorer import Scorer
from tensorboardX import SummaryWriter
import getpass
import datetime
import numpy as np

parser = argparse.ArgumentParser(description='A2C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')

parser.add_argument('--exp_name', type=str, default='vac')
parser.add_argument('--render', action='store_true')
parser.add_argument('--discount', type=float, default=1.0)
parser.add_argument('--n_iter', '-n', type=int, default=100)
parser.add_argument('--batch_size', '-b', type=int, default=32)
parser.add_argument('--ep_len', '-ep', type=float, default=10) ## Correspond to t_{max}
parser.add_argument('--actor_learning_rate', '-lr', type=float, default=5e-3)
parser.add_argument('--critic_learning_rate', '-clr', type=float, default=5e-3)
parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
parser.add_argument('--n_experiments', '-e', type=int, default=1)
parser.add_argument('--actor_n_layers', '-l', type=int, default=2)
parser.add_argument('--critic_n_layers', '-cl', type=int)
parser.add_argument('--size', '-s', type=int, default=64)

parser.add_argument("--embedding_dim", help="Embedding dimension before mapping to one-dimensional score", type=int, default = 1000)
parser.add_argument("--initial_parameters", help="Path to initial parameter file", type=str, default="alexnet.npy")
parser.add_argument("--ranking_loss", help="Type of ranking loss", type=str, choices=['ranknet', 'svm'], default='svm')
parser.add_argument("--snapshot", help="Name of the checkpoint files", type=str, default='snapshots/model-spp-max')
parser.add_argument("--spp", help="Whether to use spatial pyramid pooling in the last layer or not", type=bool, default=True)
parser.add_argument("--pooling", help="Which pooling function to use", type=str, choices=['max', 'avg'], default='max')
parser.add_argument("--act_dim", type=int, default=14)
parser.add_argument("--ob_dim", type=int, default=2000)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--entropy_loss_weight", type=float, default=0.05)
parser.add_argument("--save_path", type=str, default='checkpoint')
parser.add_argument("--image-dir", type=str, default='/xxx/datasets/ava_dataset')
parser.add_argument("--oracle_model_pth", type=str, default="/xxx/a2rl_pytorch/snapshots/model-spp-max")
parser.add_argument("--output_dir", type=str, default='/xxx/a2rl_pytorch')
parser.add_argument("--use_tensorboard", type=bool, default=True)
parser.add_argument("--runs_path", type=str, default='')
parser.add_argument("--n_epochs", type=int, default=40000)
parser.add_argument("--dataset", type=str, default="ava_images")
parser.add_argument("--save_per_epoch", type=int, default=1000)
parser.add_argument("--image_train_list", type=str, default='/xxx/sepconv-tensorflow/train_list.txt')

def generate_run_id():
    username = getpass.getuser()

    now = datetime.datetime.now()
    date = map(str, [now.year, now.month, now.day])
    coarse_time = map(str, [now.hour, now.minute])
    fine_time = map(str, [now.second, now.microsecond])

    run_id = '_'.join(['-'.join(date), '-'.join(coarse_time),
                       username, '-'.join(fine_time)])
    return run_id

if __name__ == '__main__':
    args = parser.parse_args()

    RUN_ID = generate_run_id()
    args.model_save_path = os.path.abspath(os.path.join(args.output_dir,
                                                   'models', args.dataset, RUN_ID))
    os.makedirs(args.model_save_path)

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    scorer = Scorer(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.use_tensorboard:
        args.runs_path = os.path.join(args.output_dir, 'runs_train')
        summary_writer = SummaryWriter(args.runs_path)

    train(args, scorer, summary_writer)
