# Q-Learning Algorithm

import gym
from gym import spaces
import numpy as np

env = gym.make('MountainCar-v0')

# Getting the number of available actions in the environment
ENV_ACTION_SPACE_NUMBER = env.action_space.n

# Q-Learning settings
# The LEARNING_RATE is between 0 and 1
LEARNING_RATE = 0.1
# The DISCOUNT is between 0 and 1
DISCOUNT = 0.95
EPISODES = 25000

SHOW_EVERY = 5000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Actions space
# Number of available actions as a ser {0,1,2...}
action_space = spaces.Discrete(ENV_ACTION_SPACE_NUMBER)
action = action_space.sample()
print(
    f"Available actions: {ENV_ACTION_SPACE_NUMBER}. Choosen action: {action}")

# First always reset the env
env.reset()

print(f"high OS len: {len(env.observation_space.high)}")
print(f"low OS len:{len(env.observation_space.low)}")

# Grouping the similar states into buckets (using 20 buckets * the length of the observation space)
# If the observation space has information about velocity and position, the len will be 2, meaning a matrix of 20*20.
# So we are grouping all positions and velocities to reduce the number of possible values for the Q-table
DISCRETE_OBSERVATIO_SPACE_SIZE = [20] * len(env.observation_space.high)
discrete_observation_space_window_size = (
    env.observation_space.high - env.observation_space.low)/DISCRETE_OBSERVATIO_SPACE_SIZE
print(discrete_observation_space_window_size)

# Building the q table. It is a 20*len(env.observation_space.high)*ENV_ACTION_SPACE_NUMBER shape
q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OBSERVATIO_SPACE_SIZE + [ENV_ACTION_SPACE_NUMBER]))
print(f"q table shape: {q_table.shape}")




def get_discrete_state(state):
    '''
    Basically here we are scaling the state values by making a normalization of them
    Usually  (value - minimum)/range  is a way to normalize values
    Like imagine you have values from 5 to 8,  and minimum is 5  and range is 3, you'll get all your values from 0 to 1
    'cause 
    if value = 5    (5 -5) / 3 = 0/3 = 0
    if value = 8    (8 - 5) / 3 = 3/3 =  1
    '''
    discrete_state = (state - env.observation_space.low) / discrete_observation_space_window_size
    # We use this tuple to look up the 3 Q values for the available actions in the q-table
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    # Getting the initial state from the environment
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        print(episode)

    while not done:
        # To have more exploration and to the agent not always use the first way he figured out to find his goal
        # We make a comparison with a random value. 
        # One condition we use the q table (updated by the agent, and with the ways he explored before)
        # The other condition uses a random action existing in the action space
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        # Passing the action to environment, so the agent performs the action a gets the new state and the reward. Here also we find if the environment simulation is done
        new_state, reward, done, _ = env.step(action)

        # normalizing the state we got from the environment after performing the action
        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()

        # If simulation did not end yet after last step - update Q table
        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])
            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]
            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + \
                LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state
    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
    elif new_state[0] >= env.goal_position:
        #q_table[discrete_state + (action,)] = reward
        q_table[discrete_state + (action,)] = 0


env.close()
