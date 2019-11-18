import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch.utils.data as data
import numpy as np
from torchvision import transforms
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch import nn
import torch
import nltk
import pickle
from collections import Counter
import os
import random


df = pd.read_csv('downloaded_images.csv')


class Vocabulary(object):
	"""Simple vocabulary wrapper."""

	def __init__(self):
		self.word2idx = {}
		self.idx2word = {}
		self.idx = 0

	def add_word(self, word):
		if not word in self.word2idx:
			self.word2idx[word] = self.idx
			self.idx2word[self.idx] = word
			self.idx += 1

	def __call__(self, word):
		if not word in self.word2idx:
			return self.word2idx['<unk>']
		return self.word2idx[word]

	def __len__(self):
		return len(self.word2idx)


def build_vocab(df, threshold):
	"""Build a simple vocabulary wrapper."""
	counter = Counter()
	for i in df['Caption']:
		caption = str(i)
		tokens = nltk.tokenize.word_tokenize(caption.lower())
		counter.update(tokens)

	# If the word frequency is less than 'threshold', then the word is discarded.
	words = [word for word, cnt in counter.items() if cnt >= threshold]

	# Create a vocab wrapper and add some special tokens.
	vocab = Vocabulary()
	vocab.add_word('<pad>')
	vocab.add_word('<start>')
	vocab.add_word('<end>')
	vocab.add_word('<unk>')

	# Add the words to the vocabulary.
	for i, word in enumerate(words):
		vocab.add_word(word)
	return vocab


vocab = build_vocab(df, threshold=3)
vocab_path = 'vocab.pkl'
try:
	with open(vocab_path, 'wb') as f:
		pickle.dump(vocab, f)
except:
	print('pickle error')
print("Total vocabulary size: {}".format(len(vocab)))
print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

class Dataset(data.Dataset):
	"""Custom Dataset compatible with torch.utils.data.DataLoader."""

	def __init__(self, root, df, vocab, transform=None):
		"""Set the path for images, captions and vocabulary wrapper.

		Args:
			root: image directory.
			vocab: vocabulary wrapper.
			transform: image transformer.
		"""
		self.root = root
		self.ids = df['filename'].to_list()
		self.vocab = vocab
		self.transform = transform
		self.caption = df['Caption'].to_list()

	def __getitem__(self, index):
		"""Returns one data pair (image and caption)."""
		vocab = self.vocab
		filename = self.ids[index]
		caption = self.caption[index]

		image = Image.open(os.path.join(self.root, filename)).convert('RGB')
		if self.transform is not None:
			image = self.transform(image)

		# Convert caption (string) to word ids.
		tokens = nltk.tokenize.word_tokenize(str(caption).lower())
		caption = []
		caption.append(vocab('<start>'))
		caption.extend([vocab(token) for token in tokens])
		caption.append(vocab('<end>'))
		target = torch.Tensor(caption)
		return image, target

	def __len__(self):
		return len(self.ids)


def collate_fn(data):
	"""Creates mini-batch tensors from the list of tuples (image, caption).

	We should build custom collate_fn rather than using default collate_fn,
	because merging caption (including padding) is not supported in default.
	Args:
		data: list of tuple (image, caption).
			- image: torch tensor of shape (3, 256, 256).
			- caption: torch tensor of shape (?); variable length.
	Returns:
		images: torch tensor of shape (batch_size, 3, 256, 256).
		targets: torch tensor of shape (batch_size, padded_length).
		lengths: list; valid length for each padded caption.
	"""
	# Sort a data list by caption length (descending order).
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, captions = zip(*data)

	# Merge images (from tuple of 3D tensor to 4D tensor).
	images = torch.stack(images, 0)

	# Merge captions (from tuple of 1D tensor to 2D tensor).
	lengths = [len(cap) for cap in captions]
	targets = torch.zeros(len(captions), max(lengths)).long()
	for i, cap in enumerate(captions):
		end = lengths[i]
		targets[i, :end] = cap[:end]
	return images, targets, lengths


def get_loader(root, df, vocab, transform, batch_size, shuffle, num_workers):
	"""Returns torch.utils.data.DataLoader for custom coco dataset."""
	data = Dataset(root=root,
				   vocab=vocab,
				   df=df,
				   transform=transform)
	print('dataset built!')
	# This will return (images, captions, lengths) for each iteration.
	# images: a tensor of shape (batch_size, 3, 224, 224).
	# captions: a tensor of shape (batch_size, padded_length).
	# lengths: a list indicating valid length for each caption. length is (batch_size).
	data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, num_workers=num_workers,	collate_fn=collate_fn)
	print('data loader built!')
	return data_loader


class EncoderCNN(nn.Module):
	def __init__(self, embed_size):
		"""Load the pretrained ResNet-152 and replace top fc layer."""
		super(EncoderCNN, self).__init__()
		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-1]  # delete the last fc layer.
		self.resnet = nn.Sequential(*modules)
		self.linear = nn.Linear(resnet.fc.in_features, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

	def forward(self, images):
		"""Extract feature vectors from input images."""
		with torch.no_grad():
			features = self.resnet(images)
		features = features.reshape(features.size(0), -1)
		features = self.bn(self.linear(features))
		return features


class DecoderRNN(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
		"""Set the hyper-parameters and build the layers."""
		super(DecoderRNN, self).__init__()
		self.embed = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, vocab_size)
		self.max_seg_length = max_seq_length

	def forward(self, features, captions, lengths):
		"""Decode image feature vectors and generates captions."""
		embeddings = self.embed(captions)
		embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
		packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
		hiddens, _ = self.lstm(packed)
		outputs = self.linear(hiddens[0])
		return outputs

	def sample(self, features, states=None):
		"""Generate captions for given image features using greedy search."""
		sampled_ids = []
		inputs = features.unsqueeze(1)
		for i in range(self.max_seg_length):
			hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
			outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
			_, predicted = outputs.max(1)  # predicted: (batch_size)
			sampled_ids.append(predicted)
			inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
			inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
		sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
		return sampled_ids


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = EncoderCNN(256).to(device)
decoder = DecoderRNN(256, 512, len(vocab), 1).to(device)


def run_fcn():
	torch.multiprocessing.freeze_support()
	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)
	# Create model directory
	if not os.path.exists('./models/'):
		os.makedirs('./models/')

	# Image preprocessing, normalization for the pretrained resnet
	transform = transforms.Compose([
		transforms.RandomCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406),
							(0.229, 0.224, 0.225))])
	# Load vocabulary wrapper
	# with open('./vocab.pkl', 'rb') as f:
	# 	vocab = pickle.load(f)
	# print('vocab loaded!')
	# Build data loader
	data_loader = get_loader('./Images/', df, vocab,
							transform, batch_size=64,
							shuffle=True, num_workers=0)
	# Build the models
	# encoder = EncoderCNN(256).to(device)
	# decoder = DecoderRNN(256, 512, len(vocab), 1).to(device)
	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
	optimizer = torch.optim.Adam(params, lr=0.001)

	# Train the models
	total_step = len(data_loader)
	for epoch in range(30):
		print(f'Epoch: {epoch}')
		for i, (images, captions, lengths) in enumerate(data_loader):
			# Set mini-batch dataset
			images = images.to(device)
			captions = captions.to(device)
			targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

			# Forward, backward and optimize
			features = encoder(images)
			outputs = decoder(features, captions, lengths)
			loss = criterion(outputs, targets)
			decoder.zero_grad()
			encoder.zero_grad()
			loss.backward()
			optimizer.step()

			# Print log info
			if i % 10 == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
					.format(epoch, 5, i, total_step, loss.item(), np.exp(loss.item())))

			# Save the model checkpoints
			if (i % (total_step - 1) == 0) & (i != 0):
				torch.save(decoder.state_dict(), os.path.join(
					'./models/', 'decoder-{}-{}.ckpt'.format(epoch, i)))
				torch.save(encoder.state_dict(), os.path.join(
					'./models/', 'encoder-{}-{}.ckpt'.format(epoch, i)))


def eval_fcn(image='./Images/Image_17884.jpg'):

	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	decoder_path = 'models/decoder-29-489.ckpt'
	encoder_path = 'models/encoder-29-489.ckpt'

	def load_image(image_path, transform=None):
		image = Image.open(image_path)
		image = image.resize([224, 224], Image.LANCZOS)

		if transform is not None:
			image = transform(image).unsqueeze(0)

		return image

	# Image preprocessing
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406),
		                     (0.229, 0.224, 0.225))])

	# Load vocabulary wrapper
	with open(vocab_path, 'rb') as f:
		vocab = pickle.load(f)

	# Build models
	encoder = EncoderCNN(256).eval()  # eval mode (batchnorm uses moving mean/variance)
	decoder = DecoderRNN(256, 512, len(vocab), 1)
	encoder = encoder.to(device)
	decoder = decoder.to(device)

	# Load the trained model parameters
	encoder.load_state_dict(torch.load(encoder_path))
	decoder.load_state_dict(torch.load(decoder_path))

	# Prepare an image
	image2 = load_image(image, transform)
	image_tensor = image2.to(device)

	# Generate an caption from the image
	feature = encoder(image_tensor)
	sampled_ids = decoder.sample(feature)
	sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)
	sampled_ids = sampled_ids.tolist()
	# Convert word_ids to words
	sampled_caption = []
	for word_id in sampled_ids:
		word = vocab.idx2word[word_id]
		sampled_caption.append(word)
		if word == '<end>':
			break
	sentence = ' '.join(sampled_caption)

	# Print out the image and the generated caption
	print(sentence)
	image = Image.open(image)
	plt.imshow(np.asarray(image))
	return sentence

def eval_fcn_2():
	for i in range(8):
		file = random.choice(os.listdir('Images/'))
		caption = eval_fcn('Images/'+file)
		plt.title(str(caption))
		plt.imshow(Image.open(f'Images/{file}'))
		plt.show()


if __name__ == '__main__':
	run_fcn()
	eval_fcn_2()

