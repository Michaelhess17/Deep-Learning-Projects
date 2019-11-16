from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.models import load_model
import gym
import cv2

episodes = 100000
BATCH_SIZE = 64
SAVE_EVERY = 50
MIN_MEM_SIZE = 20000
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
        self.gamma = 0.999  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(2, 2)))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(2, 2)))
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
        self.model.fit(np.array(states).reshape((BATCH_SIZE, inx, iny, c)), np.array(target_fs).reshape(BATCH_SIZE, -1),
                       epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        # serialize model to JSON
        # model_json = self.model.to_json()
        # with open("model.json", "w") as json_file:
        # 	json_file.write(model_json)
        # # serialize weights to HDF5
        # self.model.save_weights("model.h5")
        # print("Saved model to disk")
        self.model.save('model_mario.h5')

    def load_model(self):
        # self.model = self._build_model()
        self.model = load_model('model_mario.h5')


#
# Let’s Train the Agent
# The training part is even shorter. I’ll explain in the comments.
resume = True
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    state = env.reset()
    inx, iny, c = state.shape
    inx = inx // 8
    iny = iny // 8
    c = c // 3
    agent = DQNAgent((inx, iny, c), env.action_space.n)
    if resume:
        agent.load_model()
    ep_rewards = []
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
        frame = 0
        max_current_score = 0
        counter = 0
        while not done:
            # turn this on if you want to render
            # env.render()
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = cv2.resize(next_state, (inx, iny))
            next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
            # next_state = np.reshape(next_state, [1, 4])
            # Remember the previous state, action, reward, and done
            ep_reward += reward
            if ep_reward > max_current_score:
                counter = 0
                max_current_score = ep_reward
            else:
                counter += 1
                reward -= 1
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if counter == 250:
                reward -= 10
                done = True
            agent.remember(state, action, reward, next_state, done)
            if done:
                ep_rewards.append(ep_reward)
                a = min(100, len(ep_rewards))
                # print the score and break out of the loop
                print("episode: {}/{}, ep_score: {}, 100 episode trailing score: {}"
                      .format(e, episodes, ep_reward, np.mean(ep_rewards[-a:])))
            if (not frame % 25) & (len(agent.memory) > MIN_MEM_SIZE):
                agent.replay(BATCH_SIZE)
            frame += 1
        # train the agent with the experience of the episode
        # agent.replay(BATCH_SIZE)
        if not e % SAVE_EVERY:
            agent.save_model()
