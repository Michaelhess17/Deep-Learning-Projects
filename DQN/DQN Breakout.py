from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import random
import gym
import cv2
episodes = 100000
BATCH_SIZE = 64
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.compat.v1.disable_eager_execution()
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500000)
        self.gamma = 0.999    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(2,2)))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(2,2)))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.reshape(state, (1, inx, iny, c)).astype(
                'float16')
            state = state // 8
            act_values = self.model.predict(state)
            return np.argmax(act_values)  # returns action
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        next_states = []
        targets = []
        target_fs = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            state = np.reshape(state, (1, inx, iny, c)).astype(
                'float16')
            state = state // 8
            next_state = np.reshape(next_state, (1, inx, iny, c)).astype('float16')
            next_state = next_state // 8
            states.append(state)
            next_states.append(next_state)
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state))
            targets.append(target)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            target_fs.append(target_f)
        self.model.fit(np.array(states).reshape((BATCH_SIZE, inx, iny, c)), np.array(target_fs).reshape(BATCH_SIZE, -1), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#
# Let’s Train the Agent
# The training part is even shorter. I’ll explain in the comments.

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('Breakout-v0')
    state = env.reset()
    inx, iny, c = state.shape
    inx = inx // 8
    iny = iny // 8
    c = c // 3
    agent = DQNAgent((inx, iny, c), env.action_space.n)
    # Iterate the game
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()
        state = cv2.resize(state, (inx, iny))
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        # state = np.reshape(state, [1, 4])
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        ep_reward = 0
        done = False
        while not done:
            # turn this on if you want to render
            env.render()
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = cv2.resize(next_state, (inx, iny))
            next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
            # next_state = np.reshape(next_state, [1, 4])
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            ep_reward += reward
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, ep_reward))
                break
        # train the agent with the experience of the episode
        agent.replay(BATCH_SIZE)