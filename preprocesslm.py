from __future__ import print_function
import numpy as np
import gensim
import string
from lstmnumpy import lstmnumpy

print('\nFetching the text...')

print('\nPreparing the sentences...')

bptt = 40
embedding_size = 64
with open('C:/Users/ANJALI/Desktop/arxiv_abstracts.txt') as f:
    docs = f.readlines()
    sentences = [[word for word in doc.lower().translate(string.punctuation).split()[:bptt]] for doc in docs]
print('Num sentences:', len(sentences))

print('\nTraining word2vec...')
word_model = gensim.models.Word2Vec(sentences, size=embedding_size, min_count=1, window=5, iter=100)
embedding_matrix = word_model.wv.syn0
print('Result embedding shape:', embedding_matrix.shape)


def word2idx(word):
  return word_model.wv.vocab[word].index
def idx2word(idx):
  return word_model.wv.index2word[idx]

print('\nPreparing the data for LSTM...')

train_x = np.zeros([len(sentences), embedding_size, bptt])
train_y = np.zeros([len(sentences)])
for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence[:-1]):
      for j in range(0,embedding_size):
          train_x[i,j,t] = word_model.wv[word][j]
  train_y[i] = word2idx(sentence[-1])
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)

print(train_x[10,:,:].shape)

trainsample = train_x[10,:,:]


layer1 = lstmnumpy(hidden_size = 64, embedding_size = embedding_size, bptt = bptt, activation = 'sigmoid', wordmodel = word_model)
layer2 = lstmnumpy(hidden_size = 64, embedding_size = embedding_size, bptt = bptt, activation = 'sigmoid', wordmodel = word_model)
layer3 = lstmnumpy(hidden_size = 64, embedding_size = embedding_size, bptt = bptt, activation = 'softmax', last = True, wordmodel = word_model)

A = trainsample
cache = {}
model = [layer1, layer2, layer3]
for layer in model:
    Aprev = A
    A, cache = layer.forward_sequence(Aprev, train_y[10])
    

#cachenew = cache
#for t in reversed(range(len(model))):
    
    
print(A)
    
    
