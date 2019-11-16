from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, save_model
import random
import gym


# TODO double q network
# TODO OA noise class
# TODO prioritized replay buffer
# TODO dueling q network <- will be hard with Keras
# TODO make network automatically detect observation space type, and build either MLP of ConvNet

tf.compat.v1.disable_eager_execution()
episodes = 100000
BATCH_SIZE = 64
MIN_MEM = 5000
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_v2_behavior()

# Deep Q-learning Agent
class DQNAgent:
	learning_rate: float

	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=75_000)
		self.gamma = 0.99  # discount rate
		self.epsilon = 0.2  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self._build_model()

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(32, input_dim=inx[0], activation='relu'))
		model.add(Dense(16, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		else:
			act_values = self.model.predict(state)
			return np.argmax(act_values)  # returns action


	def replay(self, batch_size, epoch):
		minibatch = random.sample(self.memory, batch_size)
		states = []
		next_states = []
		targets = []
		target_fs = []
		for state, action, reward, next_state, done in minibatch:
			target = reward
			state = state.reshape((1, inx[0]))
			states.append(state)
			next_state = next_state.reshape((1, inx[0]))
			next_states.append(next_state)
			if not done:
				target = reward + self.gamma * \
						np.amax(self.model.predict(next_state))
			targets.append(target)
			target_f = self.model.predict(state)
			target_f[0][action] = target
			target_fs.append(target_f)
		self.model.fit(np.array(states).reshape((-1, inx[0])), np.array(target_fs).reshape((-1, self.action_size)),
																	epochs=epoch, verbose=0)

	def save_model(self):
		# serialize model to JSON
		# model_json = self.model.to_json()
		# with open("model.json", "w") as json_file:
		# 	json_file.write(model_json)
		# # serialize weights to HDF5
		# self.model.save_weights("model.h5")
		# print("Saved model to disk")
		self.model.save('cartpole_2.h5')

	def load_model(self):
		# self.model = self._build_model()
		self.model = load_model('cartpole_2.h5')


#
# Let’s Train the Agent
# The training part is even shorter. I’ll explain in the comments.
resume = True

if __name__ == "__main__":
	# initialize gym environment and the agent
	env = gym.make('CartPole-v1')
	inx = env.observation_space.shape
	agent = DQNAgent(env.observation_space.shape, env.action_space.n)
	if resume:
		agent.load_model()
	# Iterate the game
	ep_rewards = []
	for e in range(episodes):
		# reset state in the beginning of each game
		state = env.reset()
		state = state // 255
		state = state.reshape((1, inx[0]))
		# state = np.reshape(state, [1, 4])
		# time_t represents each frame of the game
		# Our goal is to keep the pole upright as long as possible until score of 500
		# the more time_t the more score
		ep_reward = 0
		done = False
		frames = 0
		while not done:
			# turn this on if you want to render
			# env.render()
			# Decide action
			action = agent.act(state)
			# Advance the game to the next frame based on the action.
			# Reward is 1 for every frame the pole survived
			next_state, reward, done, _ = env.step(action)
			next_state = next_state // 255
			next_state = next_state.reshape((1, inx[0]))
			# next_state = np.reshape(next_state, [1, 4])
			# Remember the previous state, action, reward, and done
			agent.remember(state, action, reward, next_state, done)
			ep_reward += reward
			# make next_state the new current state for the next frame.
			state = next_state
			# done becomes True when the game ends
			# ex) The agent drops the pole
			if done:
				ep_rewards.append(ep_reward)
				a = min(50, len(agent.memory))
				print("episode: {}/{} // ep_score: {} // average trailing score: {}"
										.format(e, 50, ep_reward, int(np.mean(ep_rewards[-a:]))))
			frames += 1
		# train the agent with the experience of the episode
		if len(agent.memory) > MIN_MEM:
			epochs = min(frames, 100)
			for i in range(epochs):
				agent.replay(BATCH_SIZE, 1)
		if not e % 50:
			agent.save_model()
