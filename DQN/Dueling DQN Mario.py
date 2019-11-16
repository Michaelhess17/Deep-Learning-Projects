from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten, Lambda, Multiply, Input
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.models import load_model
import keras.backend as K
import gym
import cv2
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

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
        self.memory = deque(maxlen=500_000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.episodes = 100_000
        self.save_every = 50
        self.min_mem_size = 20_000
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        self.update_network_every = 10
        self.dummy_input = np.zeros((1, self.action_size))
        self.dummy_batch = np.zeros((self.batch_size, self.action_size))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_layer = Input(shape=(inx, iny, c))
        conv1 = Conv2D(32, kernel_size=(2, 2), activation='relu')(input_layer)
        action_one_hot = Input(shape=(self.action_size,))
        x = MaxPooling2D()(conv1)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=(2, 2))(x)
        x = MaxPooling2D()(x)
        x = BatchNormalization()(x)
        flat_features = Flatten()(x)
        x = Dense(128, activation='relu')(flat_features)
        q_value_prediction = Dense(self.action_size, activation='linear')(x)
        # Dueling Network
        # Q = Value of state + (Value of Action - Mean of all action value)
        x_2 = Dense(512, activation='relu')(flat_features)
        state_value_prediction = Dense(1)(x_2)
        q_value_prediction = Lambda(lambda x: x[0] - K.mean(x[0]) + x[1],
                                       output_shape=(self.action_size,))([q_value_prediction, state_value_prediction])

        select_q_value_of_action = Multiply()([q_value_prediction, action_one_hot])
        target_q_value = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), output_shape=self.lambda_out_shape)(
            select_q_value_of_action)

        model = Model(inputs=[input_layer, action_one_hot], outputs=[q_value_prediction, target_q_value])
        model.compile(loss=['mse','mse'],loss_weights=[0.0,1.0],
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.reshape(state, (1, inx, iny, c)).astype(
                'float32')
            state = state // 8
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
            state_batch.append(data[0].reshape((inx, iny, c)))
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3].reshape((inx, iny, c)))
            terminal_batch.append(data[4])

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

        self.q_network.train_on_batch([self.list2np(state_batch), a_one_hot], [self.dummy_batch, y_batch])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def save_model(self):
        # serialize model to JSON
        # model_json = self.model.to_json()
        # with open("model.json", "w") as json_file:
        # 	json_file.write(model_json)
        # # serialize weights to HDF5
        # self.model.save_weights("model.h5")
        # print("Saved model to disk")
        self.q_network.save('model_mario_dueling.h5')

    def load_model(self):
        # self.model = self._build_model()
        self.q_network = load_model('model_mario_dueling.h5')

    def lambda_out_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = 1
        return tuple(shape)

    def list2np(self, in_list):
        return np.float32(np.array(in_list))


#
# Let’s Train the Agent
# The training part is even shorter. I’ll explain in the comments.
resume = False

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
    for e in range(agent.episodes):
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
                      .format(e, agent.episodes, ep_reward, int(np.mean(ep_rewards[-a:]))))
            if (not frame % 25) & (len(agent.memory) > agent.min_mem_size):
                agent.replay()
            frame += 1
        # train the agent with the experience of the episode
        if not e % agent.save_every:
            agent.save_model()
        if not e % agent.update_network_every:
            agent.update_target_network()
