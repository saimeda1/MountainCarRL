# Q-Learning with MountainCar-v0

This project demonstrates the implementation of Q-Learning for the MountainCar-v0 environment using the Gymnasium library.

## Overview

MountainCar-v0 is a classic control problem where an underpowered car must drive up a steep mountain. The goal is to reach the flag at the top right of the mountain. However, the car's engine is not strong enough to simply drive up the mountain in a single pass. The car must build up momentum by driving back and forth.

This implementation uses Q-Learning, a model-free reinforcement learning algorithm, to train the agent to solve the MountainCar-v0 environment.

## Dependencies

- numpy
- gymnasium
- matplotlib

You can install these dependencies using pip:

```bash
pip install numpy gymnasium matplotlib
```

### Code Explanation

## Importing Libraries
```python

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
```


## Environment Setup
```python
env = gym.make('MountainCar-v0', render_mode='rgb_array')
Q-Learning Parameters
```
## Q-Learning Parameters
```python
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
num_episodes = 2000
Discretizing the Observation Space
```
### Helper Functions

## Discretize State
```python
state_bins = [20, 20]  # Number of bins per state dimension
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bins = [np.linspace(b[0], b[1], num) for b, num in zip(state_bounds, state_bins)]
Initializing the Q-Table
```

## Initializing the Q-Table
```python
q_table = np.zeros((20, 20, env.action_space.n))
Helper Functions
Discretize State
```

## Discretize State
```python
def discretize_state(state):
    state_idx = []
    for i in range(len(state)):
        state_idx.append(np.digitize(state[i], state_bins[i]) - 1)
    return tuple(state_idx)
Choose Action
```

## Choose Action
```python
def choose_action(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])
Update Q-Table
```

## Update Q-Table
```python
def update_q_table(state, action, reward, next_state, done):
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + (1 - done) * gamma * q_table[next_state][best_next_action]
    td_error = td_target - q_table[state][action]
    q_table[state][action] += alpha * td_error
Training the Agent
```

## Training the Agent
```python
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
```

## Plotting Training Rewards
```pyhton
plt.plot(reward_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards')
plt.show()
Evaluating the Agent
```

## Evaluating the Agent
```python
state = discretize_state(env.reset()[0])
done = False
while not done:
    env.render()
    action = choose_action(state)
    next_state, _, done, _, _ = env.step(action)
    next_state = discretize_state(next_state)
    state = next_state
env.close()
```

Results

Environment Visualization


Training Rewards Plot
