import numpy as np
import keras as K
import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model

from keras.layers import *
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
np.random.seed(1)
print(tf.__version__)

df = pd.read_csv("Gungor_2018_VictorianAuthorAttribution_data-train.csv")

df['author'].unique()


# p = np.random.permutation(df['author'].unique())
p = df['author'].unique()
trainclass = p[:40]
fewclass = p[40:]

trainingclasses = df[df['author'].isin(trainclass)]

fewclasses = df[df['author'].isin(fewclass)]

x = np.array(trainingclasses['text'])
y = np.array(pd.factorize(trainingclasses['author'])[0])


max_words = 20000
max_review_len = 1000
                
# tokenizer = Tokenizer(20000)
# tokenizer.fit_on_texts(x)
# sequences = tokenizer.texts_to_sequences(x)
# word_index = tokenizer.word_index

# train_x, test_x, train_y, test_y = train_test_split(sequences, y, test_size=0.10, random_state=42)
# train_x = K.preprocessing.sequence.pad_sequences(train_x,
#   truncating='pre', padding='pre', maxlen=max_review_len)  # pad and chop!
# test_x = K.preprocessing.sequence.pad_sequences(test_x,
#   truncating='pre', padding='pre', maxlen=max_review_len)

# embeddings_index = {}
# f = open(os.path.join('./GLOVE_DIR', 'glove.6B.50d.txt'))
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()

# print('Found %s word vectors.' % len(embeddings_index))


# embedding_matrix = np.zeros((len(word_index) + 1, 50))

# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector

# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle)
# with open('dataset.pickle', 'wb') as handle:
#     pickle.dump((train_x, test_x, train_y, test_y), handle)
# with open('embedding_matrix.pickle', 'wb') as handle:
#     pickle.dump(embedding_matrix, handle)

with open('embedding_matrix.pickle', 'rb') as handle:
    embedding_matrix = pickle.load(handle)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('dataset.pickle', 'rb') as handle:
    train_x, test_x, train_y, test_y = pickle.load(handle)  
word_index = tokenizer.word_index
     

embedding_layer = Embedding(len(word_index) + 1,
                            50,
                            weights=[embedding_matrix],
                            input_length=max_review_len,
                            trainable=False)


# train_x, test_x, train_y, test_y = train_test_split(sequences,
#                                                     y, test_size=0.33,
#                                                     random_state=42)

layer_options = [[(16,1), (16,2), (16,3), (16,3), (16,3)],
				 [(32,1), (16,2), (16,3), (16,3), (16,3)]]
learning_rate_options = [0.01, 0.001]
batch_size_options = [128, 512]

options = [
	{
		'layers':l,
		'learning_rate':lr,
		'batch_size':b
	} for l in layer_options for lr in learning_rate_options for b in batch_size_options
]

print(options)

def network(layers, learning_rate, batch_size):
	input1 = Input(shape=(1000,))

	x = embedding_layer(input1)


	for filters, stride in layers: 
		x = Conv1D(filters, 3, strides=stride)(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Dropout(0.1)(x)


	feats = Flatten(name='features')(x)

	x = Dense(40)(feats)
	x = BatchNormalization()(x)
	x = Activation('softmax', name='main_output')(x)

	model = Model(inputs=input1, outputs=[x, feats])


	print(model.summary()) 


	opt = K.optimizers.Adam(lr=learning_rate)
	model.compile(opt, loss={'main_output': 'sparse_categorical_crossentropy'}, metrics=['acc'])


	# 3. train model
	bat_size = batch_size
	max_epochs = 50
	print("\nStarting training ")
	#model.fit(np.array(train_x), train_y, epochs=max_epochs,
	#  batch_size=bat_size, shuffle=True, verbose=1) 
	model.fit(np.array(train_x), train_y, epochs=max_epochs,
	  batch_size=bat_size, shuffle=True, verbose=1, validation_data=(np.array(test_x), test_y)) 
	print("Training complete \n")

for option in options:
	network(**option)

