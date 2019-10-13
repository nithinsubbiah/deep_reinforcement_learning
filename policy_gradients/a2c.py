import torch
import torch.nn as nn
import torch.nn.functional as F 
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorPolicy(nn.Module):

    def init_layer(self, l):
        if type(l) == nn.Linear:
            torch.nn.init.kaiming_normal_(l.weight)
            torch.nn.init.zeros_(l.bias)

    def __init__(self, input_size, output_size):
        
        super(Policy, self).__init__()

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
        
        super(Policy, self).__init__()

        
        self.no_states = input_size
        self.output_size = output_size

        self.critic_fc1.apply(self.init_layer)
        self.critic_fc1 = nn.Linear(self.no_states, 16)

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


class a2c():

    def __init__(self, environment, actor_policy, actor_lr, critic_policy, critic_lr, n=20):
        
        self.env = environment
        self.actor_policy = actor_policy
        self.actor_lr = actor_lr  
        self.critic_policy = critic_policy
        self.critic_lr = critic_lr
        self.n = n                             #The value of N in N-step A2C.

    def train():
        

    def generate_episode(self, render = False):

        states = []
        actions = []
        rewards = []
        log_probs = []
        state_values = []

        done = False
        state = self.env.reset().to(device)
    
        while not done:
    
            states.append(state)

            action_probs = self.actor_policy(state)
            state_value = self.critic_policy(state)

            # create a categorical distribution over the list of probabilities of actions
            m = Categorical(action_probs)
            # and sample an action using the distribution
            action = m.sample() 
            state, reward, done, info = env.step(action)

            actions.append(action)
            rewards.append(reward)
            log_probs.append(m.log_prob(action))
            state_values.append(state_value)

            if render:
                env.render()

        return 
    
            

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


if __name__ == '__main__':
    main(sys.argv)
