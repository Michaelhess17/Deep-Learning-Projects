# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import string

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
df = pd.read_csv('train_data.csv')
df2 = pd.read_csv('emotion.data')
df3 = pd.read_csv('WritingStories_recall.csv')
df2.drop('Unnamed: 0', axis=1, inplace=True)
labels = {'fun': 'joy', 'happiness': 'joy', 'hate': 'anger', 'worry': 'fear', 'anger': 'anger', 'love': 'love',
          'sadness': 'sadness', 'surprise': 'surprise'}
df['sentiment'] = df['sentiment'].map(labels)
df.columns = ['emotions', 'text']
df = pd.concat([df, df2])
df = df[~df['emotions'].isnull()]
df.emotions.value_counts().plot.bar()
plt.show()
# %%
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text
input_sentences = [clean_text(text) for text in df["text"].values.tolist()]
labels = df["emotions"].values.tolist()

word2id = dict()
label2id = dict()

max_words = 0  # maximum number of words in a sentence

# Construction of word2id dict
for sentence in input_sentences:
	for word in sentence:
		# Add words to word2id dict if not exist
		if word not in word2id:
			word2id[word] = len(word2id)
	# If length of the sentence is greater than max_words, update max_words
	if len(sentence) > max_words:
		max_words = len(sentence)

# Construction of label2id and id2label dicts
label2id = {l: i for i, l in enumerate(set(labels))}
id2label = {v: k for k, v in label2id.items()}
print(id2label)
# %%
import keras

# Encode input words and labels
X = [[word2id[word] for word in sentence] for sentence in input_sentences]
Y = [label2id[label] for label in labels]

# Apply Padding to X
from keras.preprocessing.sequence import pad_sequences

X = pad_sequences(X, max_words)

# Convert Y to numpy array
Y = keras.utils.to_categorical(Y, num_classes=len(label2id), dtype='float32')

# Print shapes
print("Shape of X: {}".format(X.shape))
print("Shape of Y: {}".format(Y.shape))
# %%
embedding_dim = 128  # The dimension of word embeddings

# Define input tensor
sequence_input = keras.Input(shape=(max_words,), dtype='int32')

# Word embedding layer
embedded_inputs = keras.layers.Embedding(len(word2id) + 1,
                                         embedding_dim,
                                         input_length=max_words)(sequence_input)

# Apply dropout to prevent overfitting
embedded_inputs = keras.layers.Dropout(0.4)(embedded_inputs)

# Apply Bidirectional LSTM over embedded inputs
lstm_outs = keras.layers.wrappers.Bidirectional(
	keras.layers.LSTM(embedding_dim, return_sequences=True)
)(embedded_inputs)

# Apply dropout to LSTM outputs to prevent overfitting
lstm_outs = keras.layers.Dropout(0.4)(lstm_outs)

# Attention Mechanism - Generate attention vectors
input_dim = int(lstm_outs.shape[2])
permuted_inputs = keras.layers.Permute((2, 1))(lstm_outs)
attention_vector = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_outs)
attention_vector = keras.layers.Reshape((max_words,))(attention_vector)
attention_vector = keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
attention_vector = keras.layers.Dropout(0.4)(attention_vector)
attention_output = keras.layers.Dot(axes=1)([lstm_outs, attention_vector])

# Last layer: fully connected with softmax activation
fc = keras.layers.Dense(embedding_dim, activation='relu')(attention_output)
output = keras.layers.Dense(len(label2id), activation='softmax')(fc)

# Finally building model
model = keras.Model(inputs=[sequence_input], outputs=output)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')

# Print model summary
model.summary()

#%%
model.fit(X, Y, epochs=2, batch_size=64, validation_split=0.2, shuffle=True)
#%%
model_with_attentions = keras.Model(inputs=model.input,
                                    outputs=[model.output,
                                             model.get_layer('attention_vec').output])
import random
import math

# Select random samples to illustrate
mask = (df['text'].str.len() > 150)
sample_text = ''
for _ in range(5):
	sample_text = sample_text + random.choice(df.loc[mask]['text'].values.tolist())
print(sample_text)

# Encode samples
tokenized_sample = clean_text(sample_text)
encoded_samples = [[word2id[word] for word in tokenized_sample]]

# Padding
encoded_samples = keras.preprocessing.sequence.pad_sequences(encoded_samples, maxlen=max_words)

# Make predictions
label_probs, attentions = model_with_attentions.predict(encoded_samples)
label_probs = {id2label[_id]: prob for (label, _id), prob in zip(label2id.items(), label_probs[0])}

# Get word attentions using attenion vector
token_attention_dic = {}
max_score = 0.0
min_score = 0.0
for token, attention_score in zip(tokenized_sample, attentions[0][-len(tokenized_sample):]):
	token_attention_dic[token] = math.sqrt(attention_score)

# VISUALIZATION
import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML


def rgb_to_hex(rgb):
	return '#%02x%02x%02x' % rgb


def attention2color(attention_score):
	r = 255 - int(attention_score * 255)
	color = rgb_to_hex((255, r, r))
	return str(color)


# Build HTML String to viualize attentions
html_text = "<hr><p style='font-size: large'><b>Text:  </b>"
for token, attention in token_attention_dic.items():
	html_text += "<span style='background-color:{};'>{} <span> ".format(attention2color(attention),
	                                                                    token)
html_text += "</p>"
# Display text enriched with attention scores
display(HTML(html_text))

# PLOT EMOTION SCORES
emotions = [label for label, _ in label_probs.items()]
scores = [score for _, score in label_probs.items()]
plt.figure(figsize=(5, 2))
plt.bar(np.arange(len(emotions)), scores, align='center', alpha=0.5,
        color=['black', 'red', 'green', 'blue', 'cyan', "purple"])
plt.xticks(np.arange(len(emotions)), emotions)
plt.ylabel('Scores')
plt.show()