import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from model import ActorCritic
from train import train
import my_optim
from scorer import Scorer
from tensorboardX import SummaryWriter
import getpass
import datetime
import numpy as np
from envs import create_crop_env
import torchvision
import skimage.transform as transform



parser = argparse.ArgumentParser(description='A2C')

parser.add_argument("--image-dir", type=str, default='/xxx/sunyasheng/datasets/data')
parser.add_argument("--model-path", type=str, default='/xxx/a2rl_pytorch/models/ava_images/2019-6-6_15-32_root_18-380442/model_119_6.347011566162109.pth')
parser.add_argument("--output_dir", type=str, default='/xxx/a2rl_pytorch')
parser.add_argument("--n-epochs", type=int, default=50)
parser.add_argument("--use-tensorboard", type=bool, default=True)
parser.add_argument("--highest-step", type=int, default=25)
parser.add_argument("--oracle_model_pth", type=str, default="/xxxx/a2rl_pytorch/snapshots/model-spp-max")
parser.add_argument("--add-image-per-epoch", type=int, default=2)

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
parser.add_argument('--batch_size', '-b', type=int, default=30)
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

def test(args):
    args.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    scorer = Scorer(args)

    if args.use_tensorboard:
        args.runs_path = os.path.join(args.output_dir, 'runs_test')
        summary_writer = SummaryWriter(args.runs_path)

    model = ActorCritic(args).to(args.device)
    model.eval()

    for epoch_id in range(args.n_epochs):
        cur_reward, used_steps, not_finish, status = run_one_epoch(args, scorer, model)
        summary_writer.add_scalar('aver_reward', cur_reward, epoch_id)
        summary_writer.add_scalar('used_steps', used_steps, epoch_id)
        if (epoch_id + 1) % args.add_image_per_epoch == 0:
            (origin_img, cropped_bbox, score_diff) = status
            # import pdb; pdb.set_trace();
            (xmin, ymin, xmax, ymax) = cropped_bbox
            cropped_img = np.ones_like(origin_img) * 255
            cropped_img[ymin:ymax, xmin:xmax, :] = origin_img[ymin:ymax, xmin:xmax, :]
            # cropped_img = transform.resize(cropped_img, (origin_img.shape[0], origin_img.shape[1]))
            [origin_img, cropped_img] = map(lambda x: x.transpose((2, 0, 1)), [origin_img, cropped_img])
            # summary_writer.add_image('origin_img {}'.format(epoch_id), origin_img, epoch_id)
            # summary_writer.add_image('cropped_img {}'.format(epoch_id), cropped_img, epoch_id)
            stacked_img = torchvision.utils.make_grid(torch.from_numpy(np.stack((origin_img, cropped_img))), nrow=1, padding=2)
            summary_writer.add_image('origin_cropped {}'.format(epoch_id), stacked_img)
            summary_writer.add_scalar('score_diff {}'.format(epoch_id), score_diff)
        print("epoch : {:03f},  aver_reward: {:03f}, used_steps: {:03d}, not_finish: {:d}".\
                    format(epoch_id, cur_reward, used_steps, int(not_finish)))

@torch.no_grad()
def run_one_epoch(args, scorer, model, summary_writer=None):
    env = create_crop_env(args, scorer)

    cx = torch.zeros(1, args.hidden_dim).to(args.device)
    hx = torch.zeros(1, args.hidden_dim).to(args.device)

    observation_np = env.reset()
    done, not_finish = False, False
    tot_reward = 0
    tot_steps = 0
    status = None

    while not done:
        observation_ts = torch.from_numpy(observation_np).to(args.device)
        value_ts, logit_ts, (hx, cx) = model((observation_ts,
                                              (hx, cx)))
        prob = F.softmax(logit_ts, dim=-1)
        action_ts = prob.multinomial(num_samples=1).detach()

        action_np = action_ts.cpu().numpy()

        observation_np, reward, done, _ = env.step(action_np)
        status = env.cur_status()

        tot_reward += reward
        tot_steps += 1
        if tot_steps > args.highest_step:
            not_finish = True
            break

    return tot_reward, tot_steps, not_finish, status

if __name__ == '__main__':
    args = parser.parse_args()
    test(args)