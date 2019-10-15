import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical
import gym
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import date
from tensorboardX import SummaryWriter
writer = SummaryWriter("runs_a2c/exp1")
date_str = date.today().strftime("%d_%m_%Y")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorPolicy(nn.Module):

    def init_layer(self, l):
        if type(l) == nn.Linear:
            torch.nn.init.kaiming_normal_(l.weight)
            torch.nn.init.zeros_(l.bias)

    def __init__(self, input_size, output_size):
        
        super(ActorPolicy, self).__init__()

        # Actor network
        self.no_states = input_size
        self.no_action = output_size
        
        self.actor_fc1 = nn.Linear(self.no_states, 16)
        self.actor_fc1.apply(self.init_layer)
        # self.fc1.bias.data.fill_(0)

        self.actor_fc2 = nn.Linear(16, 16)
        self.actor_fc2.apply(self.init_layer)
        
        self.actor_fc3 = nn.Linear(16, 16)
        self.actor_fc3.apply(self.init_layer)
        
        self.actor_fc4 = nn.Linear(16, self.no_action)
        self.actor_fc4.apply(self.init_layer)

    def forward(self, state):

        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        x = F.relu(self.actor_fc3(x))
        action_prob = F.softmax(self.actor_fc4(x)) 

        return action_prob

class CriticPolicy(nn.Module):

    def init_layer(self, l):
        if type(l) == nn.Linear:
            torch.nn.init.kaiming_normal_(l.weight)
            torch.nn.init.zeros_(l.bias)

    def __init__(self, input_size, output_size):
        
        super(CriticPolicy, self).__init__()

        
        self.no_states = input_size
        self.output_size = output_size

        self.critic_fc1 = nn.Linear(self.no_states, 16)
        self.critic_fc1.apply(self.init_layer)

        self.critic_fc2 = nn.Linear(16, 16)
        self.critic_fc2.apply(self.init_layer)
        
        self.critic_fc3 = nn.Linear(16, 16)
        self.critic_fc3.apply(self.init_layer)
        
        self.critic_fc4 = nn.Linear(16, self.output_size)
        self.critic_fc4.apply(self.init_layer)

    def forward(self, state):

        x = F.relu(self.critic_fc1(state))
        x = F.relu(self.critic_fc2(x))
        x = F.relu(self.critic_fc3(x))
        state_value = F.softmax(self.critic_fc4(x)) 

        return state_value


class A2C():

    def __init__(self, environment, actor_policy, actor_lr, critic_policy, critic_lr, num_episodes, render, n = 20, discount_factor = 0.9):
        
        self.env = environment
        self.actor_policy = actor_policy
        self.actor_lr = actor_lr  
        self.critic_policy = critic_policy
        self.critic_lr = critic_lr
        self.num_episodes = num_episodes
        self.n = n                             #The value of N in N-step A2C.
        self.render = render
        self.gamma = discount_factor

        self.actor_optimizer = torch.optim.Adam(actor_policy.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic_policy.parameters(), lr=critic_lr)

        self.test_frequency = 500
        self.summary_frequency = 100
        self.saving_frequency = 2000
        self.test_rewards = []
        self.test_rewards_stddev = []

    def generate_episode(self):

        states = []
        actions = []
        rewards = []
        log_probs = []
        state_values = []

        done = False
        state = torch.from_numpy(self.env.reset()).to(device)
    
        while not done:
    
            states.append(state)

            action_probs = self.actor_policy(state)
            state_value = self.critic_policy(state)

            # create a categorical distribution over the list of probabilities of actions
            m = Categorical(action_probs)
            # and sample an action using the distribution
            action = m.sample() 
            state, reward, done, info = self.env.step(action.item())
            
            state = torch.from_numpy(state).to(device)

            reward = torch.from_numpy(np.asarray(reward, dtype=np.float32)).to(device)
            state_value = state_value.to(device)

            actions.append(action)
            rewards.append(reward)
            log_probs.append(m.log_prob(action))
            state_values.append(state_value)

            if self.render:
                env.render()
        
        return torch.stack(log_probs), torch.stack(state_values), torch.squeeze(torch.stack(rewards)), torch.stack(actions), torch.stack(states)  

    def train(self):
        
        policy_losses = []
        value_losses = []

        for episode in range(self.num_episodes):
            
            log_probs, state_values, rewards, actions, states = self.generate_episode()

            episode_length = len(states)

            # List to store the N-step return for each step in trajectory
            N_step_returns = []

            # Looping through each step of the episode length from the last
            for t in reversed(range(episode_length)):
                
                # Find the state value from the nth state
                if (t+self.n) >= episode_length:
                    V_end = 0
                else:
                    V_end = state_values[t+self.n].item()

                G = 0
                # Find the utility function until nth state
                for k in range(self.n):
                    if (t+k) < episode_length:
                        G = G + np.power(self.gamma, k) * rewards[t+k]

                # Sum both returns till nth step and value of the (n+1)th step
                n_step_return = np.power(self.gamma, self.n) * V_end + G 
                N_step_returns.append(n_step_return)

            # Reverses list to denote trajectory from t to episode length
            N_step_returns.reverse()

            N_step_returns = torch.stack(N_step_returns)
            state_values = torch.squeeze(state_values)

            # No state_value grad needed for actor update
            with torch.no_grad():
                advantage_actor = N_step_returns - state_values
            actor_loss = (-log_probs*advantage_actor).mean()

            advantage_critic = N_step_returns - state_values
            critic_loss = advantage_critic.pow(2).mean()

            total_loss = actor_loss + critic_loss            

            self.actor_policy.train()
            self.critic_policy.train()     

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            writer.add_scalar('lunarlander_a2c/reward_episode', np.array(torch.sum(rewards).item()), episode)
            
            # Summaries
            if(episode%self.summary_frequency == 0):
                # meme reference
                print ("Is this LOSS? : " + str(total_loss.item()) + " Reward: " + str(torch.sum(rewards).item()))
            
            if(episode%self.saving_frequency == 0):
                # saving model
                print ("Saving Model")
                torch.save(self.actor_policy, "./runs_a2c/a2c_actor_" + str(episode) + "_" + date_str)
                torch.save(self.critic_policy, "./runs_a2c/a2c_critic_" + str(episode) + "_" + date_str)

            if(episode%self.test_frequency == 0):
                print ("Testing")
                self.test(episode)

    def test(self, episode_no):
        
        self.actor_policy.eval()

        test_reward_arr = []
        test_reward = torch.Tensor([0])
        
        for episode_count in range(100):
            _, _, rewards, _, _ = self.generate_episode()
            test_reward += torch.sum(rewards)
            test_reward_arr.append(torch.sum(rewards))

        test_reward = test_reward/100

        # standard deviation
        test_reward_arr = torch.stack(test_reward_arr)
        test_reward_std_dev = test_reward_arr.std().item()

        writer.add_scalar('lunarlander_a2c/cumulative_average_reward', np.array(test_reward.item()), episode_no)

        self.test_rewards.append(test_reward.item())
        self.test_rewards_stddev.append(test_reward_std_dev)
        range_x_axis = np.arange(0, episode_no+1, self.test_frequency)

        plt.errorbar(range_x_axis, self.test_rewards, self.test_rewards_stddev)
        plt.xlabel('Episode number')
        plt.ylabel('Cumulative Test Rewards / Standard Deviation')
        plt.savefig("./runs_a2c/Test_plot_%d.png"%episode_no)

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    state_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n

    actor = ActorPolicy(state_space_size, action_space_size)
    critic = CriticPolicy(state_space_size, 1)

    a2c = A2C(env, actor, lr, critic, critic_lr, num_episodes,render, n)

    a2c.train()

if __name__ == '__main__':
    main(sys.argv)
