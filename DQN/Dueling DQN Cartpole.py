from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten, Lambda, Multiply, Input
from tensorflow.keras.optimizers import SGD
import random
from tensorflow.keras.models import load_model
import keras.backend as K
import gym
from gym import wrappers
import cv2
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from customLayers import LambdaLayer

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.compat.v1.disable_eager_execution()
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Dueling Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50_000)
        self.gamma = 0.98  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9975
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.episodes = 25_000
        self.save_every = 50
        self.min_mem_size = 5_000
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        self.update_network_every = 10
        self.dummy_input = np.zeros((1, self.action_size))
        self.dummy_batch = np.zeros((self.batch_size, self.action_size))
        self.replay_counter = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_layer = Input(shape=(self.state_size))
        x = Dense(64)(input_layer)
        action_one_hot = Input(shape=(self.action_size,))
        x = Dense(32, activation='relu')(x)
        flat_features = Flatten()(x)
        x = Dense(16, activation='relu')(flat_features)
        q_value_prediction = Dense(self.action_size, activation='linear')(x)
        # Dueling Network
        # Q = Value of state + (Value of Action - Mean of all action value)
        x_2 = Dense(16, activation='relu')(flat_features)
        state_value_prediction = Dense(1)(x_2)
        q_value_prediction = Lambda(lambda x: x[0] - K.mean(x[0]) + x[1],
                                    output_shape=(self.action_size,))([q_value_prediction, state_value_prediction])

        select_q_value_of_action = Multiply()([q_value_prediction, action_one_hot])
        target_q_value = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), output_shape=self.lambda_out_shape)(
            select_q_value_of_action)

        model = Model(inputs=[input_layer, action_one_hot], outputs=[q_value_prediction, target_q_value])
        model.compile(loss=['mse', 'mse'],loss_weights=[0.00, 1.0],
                      optimizer=SGD(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.target_network.predict([state, self.dummy_input])
            return np.argmax(act_values[0])  # returns action
    def act_greedily(self, state):
        act_values = self.target_network.predict([state, self.dummy_input])
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.memory, self.batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])
        state_batch = np.squeeze(state_batch)
        next_state_batch = np.squeeze(next_state_batch)
        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0
        # Q value from target network
        target_q_values_batch = self.target_network.predict([self.list2np(next_state_batch), self.dummy_batch])[0]

        # create Y batch depends on dqn or ddqn
        next_action_batch = np.argmax(self.q_network.predict([self.list2np(next_state_batch), self.dummy_batch])[0],
                                          axis=-1)
        for i in range(self.batch_size):
            y_batch.append(reward_batch[i] + (1 - terminal_batch[i]) * self.gamma * target_q_values_batch[i][
                next_action_batch[i]])
        y_batch = self.list2np(y_batch)

        a_one_hot = np.zeros((self.batch_size, self.action_size))
        for idx, ac in enumerate(action_batch):
            a_one_hot[idx, ac] = 1.0

        loss = self.q_network.train_on_batch([self.list2np(state_batch), a_one_hot], [self.dummy_batch, y_batch])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.replay_counter += 1
        return loss
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def save_model(self):
        self.q_network.save_weights('cartpole_dueling.h5')

    def load_model(self):
        self.q_network.load_weights('cartpole_dueling.h5')

    def lambda_out_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = 1
        return tuple(shape)

    def list2np(self, in_list):
        return np.float32(np.array(in_list))


#
# Let’s Train the Agent
# The training part is even shorter. I’ll explain in the comments.
resume = True
solved = False
if __name__ == "__main__":
    if not solved:
        # initialize gym environment and the agent
        env = gym.make('CartPole-v1')
        agent = DQNAgent(env.observation_space.shape, env.action_space.n)
        if resume:
            agent.load_model()
        ep_rewards = [0]
        # Iterate the game
        for e in range(agent.episodes):
            # reset state in the beginning of each game
            state = env.reset()
            state = np.reshape(state, (1, env.observation_space.shape[0]))
            ep_reward = 0
            done = False
            frame = 0
            while (not done) & (not solved):
                # turn this on if you want to render
                # env.render()
                # Decide action
                action = agent.act(state)
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, (1, env.observation_space.shape[0]))
                # Remember the previous state, action, reward, and done
                ep_reward += reward
                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                # ex) The agent drops the pole
                agent.remember(state, action, reward, next_state, done)
                if done:
                    if len(agent.memory) > agent.min_mem_size:
                        loss = agent.replay()
                        total_loss = loss[0]
                        lambda_loss = loss[1]
                        lambda1_loss = loss[2]
                    else:
                        total_loss = 0
                        lambda_loss = 0
                        lambda1_loss = 0
                    ep_rewards.append(ep_reward)
                    a = min(100, len(ep_rewards))
                    # print the score and break out of the loop
                    print("episode: {}/{} // ep_score: {} // 100 episode trailing score: {} // loss: {} // lambda loss: {}  // lambda 1 loss: {}"
                          .format(e, agent.episodes, ep_reward, int(np.mean(ep_rewards[-a:])), total_loss, lambda_loss, lambda1_loss))
                if (not frame % 1) & (len(agent.memory) > agent.min_mem_size):
                    agent.replay()
                frame += 1
                if int(np.mean(ep_rewards[-50:])) > 450:
                    solved = True
                if not agent.replay_counter % agent.update_network_every:
                    agent.update_target_network()
                    agent.replay_counter = 0
            # train the agent with the experience of the episode
            if (not e % agent.save_every) or solved:
                agent.save_model()
            if solved:
                env = gym.make('CartPole-v0')
                env = wrappers.Monitor(env, 'CartPole-v0', force=True)
                ep_rewards = []
                for e in range(50):
                    done = False
                    frame = 0
                    state = env.reset()
                    ep_reward = 0
                    while not done:
                        env.render()
                        action = agent.act_greedily(state)
                        next_state, reward, done, _ = env.step(action)
                        ep_reward += reward
                        state = next_state
                    ep_rewards.append(ep_reward)
                    # print the score and break out of the loop
                    print("episode: {}/{} // ep_score: {} // average trailing score: {}"
                          .format(e, 50, ep_reward, int(np.mean(ep_rewards[-e:]))))


