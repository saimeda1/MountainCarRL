import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Create the MountainCar-v0 environment
env = gym.make('MountainCar-v0',render_mode='human')

# Q-learning parameters
alpha = 0.1  # the rate it learns at, how much q values change
gamma = 0.99  #future rewards are super important, the flag is the goal and it gets harder to go up mountain
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01 # always explores
epsilon_decay = 0.995
num_episodes = 2000

# Discretize the observation space
state_bins = [20, 20]  # Number of bins per state dimension
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bins = [np.linspace(b[0], b[1], num) for b, num in zip(state_bounds, state_bins)]

# Initialize the Q-table
q_table = np.zeros((20, 20, env.action_space.n))

def discretize_state(state):
    #function can't convert continous state so it changes it to a scalar state on it's bins
    state_idx = []
    for i in range(len(state)):
        state_idx.append(np.digitize(state[i], state_bins[i]) - 1)
    return tuple(state_idx)

def choose_action(state):
    # chooses which action from 0-2 based on my greedy policy
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

def update_q_table(state, action, reward, next_state, done):
    #updates the q values
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + (1 - done) * gamma * q_table[next_state][best_next_action]
    td_error = td_target - q_table[state][action]
    q_table[state][action] += alpha * td_error

# agent training
reward_list = []
for episode in range(num_episodes):
    state = discretize_state(env.reset()[0])
    total_reward = 0
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        update_q_table(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    reward_list.append(total_reward)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")



# Plotting the rewards
plt.plot(reward_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards')
plt.show()

# Evaluating the agent
state = discretize_state(env.reset()[0])
done = False
while not done:
    env.render()
    action = choose_action(state)
    next_state, _, done, _, _ = env.step(action)
    next_state = discretize_state(next_state)
    state = next_state

env.close()
