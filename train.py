import torch
import torch.nn.functional as F
import torch.optim as optim

from model import ActorCritic
from envs import create_crop_env
from episode import BatchEpisodes
import numpy as np
from utils.torch_utils import weighted_normalize, weighted_mean
import torch.optim as optim
import os

def train(args, scorer, summary_writer=None):
    device = args.device
    env = create_crop_env(args, scorer)

    model = ActorCritic(args).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # import pdb; pdb.set_trace();
    training_log_file = open(os.path.join(
        args.model_save_path, 'training.log'), 'w')
    validation_log_file = open(os.path.join(
        args.model_save_path, 'validation.log'), 'w')

    training_log_file.write('Epoch,Cost\n')
    validation_log_file.write('Epoch,Cost\n')

    for train_iter in range(args.n_epochs):
        episode = BatchEpisodes(batch_size=args.batch_size, gamma=args.gamma, device=device)

        for _ in range(args.batch_size):
            done = True
            observation_np = env.reset()

            observations_np, rewards_np, actions_np, hs_ts, cs_ts = [], [], [], [], []
            cx = torch.zeros(1, args.hidden_dim).to(device)
            hx = torch.zeros(1, args.hidden_dim).to(device)
            
            for step in range(args.num_steps):
                observations_np.append(observation_np[0])
                hs_ts.append(hx)
                cs_ts.append(cx)

                with torch.no_grad():
                    observation_ts = torch.from_numpy(observation_np).to(device)
                    value_ts, logit_ts, (hx, cx) = model((observation_ts,
                                                (hx, cx)))       
                    prob = F.softmax(logit_ts, dim=-1)         
                    action_ts = prob.multinomial(num_samples=1).detach()
                
                action_np = action_ts.cpu().numpy()
                actions_np.append(action_np[0][0])
                observation_np, reward_num, done, _ = env.step(action_np)
                if step == args.num_steps - 1:
                    reward_num = 0 if done else value_ts.item()
                rewards_np.append(reward_num)

                if done:
                    break

            observations_np, actions_np, rewards_np = \
                map(lambda x: np.array(x).astype(np.float32), [observations_np, actions_np, rewards_np])
            episode.append(observations_np, actions_np, rewards_np, hs_ts, cs_ts)

        log_probs = []
        values = []
        entropys = []
        for i in range(len(episode)):
            (hs_ts, cs_ts) = episode.hiddens[0][i], episode.hiddens[1][i]
            value_ts, logit_ts, (_, _) = model((episode.observations[i], (hs_ts, cs_ts)))
            prob = F.softmax(logit_ts, dim=-1)
            log_prob = F.log_softmax(logit_ts, dim=-1)
            entropy = -(log_prob * prob).sum(1)
            log_prob = log_prob.gather(1, episode.actions[i].unsqueeze(1).long())
            log_probs.append(log_prob)
            entropys.append(entropy)
            values.append(value_ts)

        log_probs_ts = torch.stack(log_probs).squeeze(2)
        values_ts = torch.stack(values).squeeze(2)
        entropys_ts = torch.stack(entropys)

        advantages_ts = episode.gae(values_ts)
        advantages_ts = weighted_normalize(advantages_ts, weights=episode.mask)
        policy_loss = - weighted_mean(log_probs_ts * advantages_ts, dim=0,
                weights=episode.mask)
        # import pdb; pdb.set_trace();
        value_loss = weighted_mean((values_ts - episode.returns).pow(2), dim=0,
                weights = episode.mask)
        entropy_loss = - weighted_mean(entropys_ts, dim=0,
                weights = episode.mask)
        
        optimizer.zero_grad()
        tot_loss = policy_loss + entropy_loss + args.value_loss_coef * value_loss
        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        print("Epoch [%2d/%2d] : Tot Loss: %5.5f, Policy Loss: %5.5f, Value Loss: %5.5f, Entropy Loss: %5.5f" %
              (train_iter, args.n_epochs, tot_loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()))
        # print("Train_iter: ", train_iter, " Total Loss: ", tot_loss.item(), " Value Loss: ", value_loss.item(), " Policy Loss: ", policy_loss.item(), "Entropy Loss: ", entropy_loss.item())
        if summary_writer:
            summary_writer.add_scalar('loss_policy', policy_loss.item(), train_iter)
            summary_writer.add_scalar('loss_value', value_loss.item(), train_iter)
            summary_writer.add_scalar('loss_entropy', entropy_loss.item(), train_iter)
            summary_writer.add_scalar('loss_tot', tot_loss.item(), train_iter)
        train_iter += 1

        if (train_iter + 1) % args.save_per_epoch == 0:
            torch.save(model.state_dict(), os.path.join(args.model_save_path,
                                                        'model_{}_{}.pth').format(train_iter, tot_loss.item()))

        training_log_file.write('{},{}\n'.format(train_iter, tot_loss.item()))
        validation_log_file.write('{},{}\n'.format(train_iter, 0))
        training_log_file.flush()
        validation_log_file.flush()

    training_log_file.close()
    validation_log_file.close()

'''
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state,
                                            (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length

            if done:
                episode_length = 0
                state = env.reset()
            
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state, (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        tot_loss = policy_loss + args.value_loss_coef * value_loss
        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        
        print("Train_iter: ", train_iter, " Total Loss: ", tot_loss.item(), " Value Loss: ", value_loss.item(), " Policy Loss: ", policy_loss.item())
        if summary_writer:
            summary_writer.add_scalar('loss_policy', policy_loss.item(), train_iter)
            summary_writer.add_scalar('loss_value', value_loss.item(), train_iter)
            summary_writer.add_scalar('loss_tot', tot_loss.item(), train_iter)
        train_iter += 1
'''