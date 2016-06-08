# -*- coding: utf-8 -*-
'''Teaches network to solve a brackets priority problem
adapted from keras examples: 
https://raw.githubusercontent.com/fchollet/keras/master/examples/addition_rnn.py
'''

from __future__ import print_function

import numpy as np

from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from six.moves import range


class CharacterTable(object):
	'''
	Given a set of characters:
	+ Encode them to a one hot integer representation
	+ Decode the one hot integer representation to their character output
	+ Decode a vector of probabilities to their character output
	'''
	def __init__(self, chars, maxlen):
		self.chars = sorted(set(chars))
		self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
		self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
		self.maxlen = maxlen

	def encode(self, C, maxlen=None):
		maxlen = maxlen if maxlen else self.maxlen
		X = np.zeros((maxlen, len(self.chars)))
		for i, c in enumerate(C):
			X[i, self.char_indices[c]] = 1
		return X

	def decode(self, X, calc_argmax=True):
		if calc_argmax:
			X = X.argmax(axis=-1)
		return ''.join(self.indices_char[x] for x in X)


class DataGenerator( object ):
	def generate( data_size, digit_max_len ):
		
		questions = []
		answers = []
		seen = set()
		
		f = lambda: ''.join( np.random.choice( list( '0123456789' ) ) for i in range( np.random.randint( 1, digit_max_len + 1) ) )
		g = lambda: np.random.choice( list( '+-*/' ) )
		h = lambda: np.random.choice( list( 'LR' ) )

		i = 0
		while i < set_size:
			n1 = f()
			n2 = f()
			n3 = f()
			
			o1 = g()
			o2 = g()
			
			q = []
			a = []
			
			if h() == 'L':
				if o1 in '*/':
					if o2 in '-+':
						q = [ n1, o1, '(', n2, o2, n3, ')', '=' ]
						a = [ n2, o2, n3, '=', o1, n1, '=' ]
					else
						q = [ n1, o1, n2, o2, n3, '=' ]
						a = [ n1, o1, n2, o2, n3, '=' ]
				else:
					q = [ n1, o1, n2, o2, n3, '=' ]
					a = [ n1, o1, n2, o2, n3, '=' ]
			else:
				if o3 in '*/':
					if o2 in '-+':
						q = [ '(', n1, o1, n2, ')', o2, n3, '=' ]
						a = [ n1, o1, n2, '=', o2, n3, '=' ]
					else
						q = [ n1, o1, n2, o2, n3, '=' ]
						a = [ n1, o1, n2, o2, n3, '=' ]
				else:
					q = [ n1, o1, n2, o2, n3, '=' ]
					a = [ n1, o1, n2, o2, n3, '=' ]
				
			s = ''.join( q )
			if s not in seen:
				questions.append( q )
				answers.append( a )
				seen.add( s )
				i += 1
				
		return questions, answers

class colors:
	ok = '\033[92m'
	fail = '\033[91m'
	close = '\033[0m'

# Parameters for the model and dataset
TRAINING_SIZE = 50000
DIGITS = 3
INVERT = True
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
MAXLEN = DIGITS + 1 + DIGITS

ctable = CharacterTable(chars, MAXLEN)

print('Generating data...')
questions, answers = DataGenerator.generate( TRAINING_SIZE, DIGITS )
print('Total addition questions:', len( questions ) )

print('Vectorization...')
X = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    X[i] = ctable.encode(sentence, maxlen=MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, maxlen=DIGITS + 1)

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over
split_at = len(X) - len(X) / 10
(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
(y_train, y_val) = (y[:split_at], y[split_at:])

print(X_train.shape)
print(y_train.shape)

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(DIGITS + 1))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = ctable.decode(rowX[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if INVERT else q)
        print('T', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        print('---')
