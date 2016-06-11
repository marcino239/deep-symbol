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


class CharacterTable( object ):
	'''
	Given a set of characters:
	+ Encode them to a one hot integer representation
	+ Decode the one hot integer representation to their character output
	+ Decode a vector of probabilities to their character output
	'''
	def __init__( self, chars, maxlen=None ):
		self.chars = sorted( set(chars) )
		self.char_indices = dict( (c, i) for i, c in enumerate( self.chars ) )
		self.indices_char = dict( (i, c) for i, c in enumerate( self.chars ) )
		self.maxlen = maxlen or len( chars )

	def encode( self, C, maxlen=None ):
		maxlen = maxlen if maxlen else self.maxlen
		X = np.zeros( ( maxlen, len( self.chars ) ), dtype=np.bool )
		for i, c in enumerate( C ):
			X[ i, self.char_indices[ c ] ] = 1
		return X

	def decode( self, X, calc_argmax=True ):
		if calc_argmax:
			X = X.argmax( axis = -1 )
		return ''.join( self.indices_char[ x ] for x in X )


class DataGenerator( object ):
	def __init__( self, train_size, test_size, number_max_len ):
		self.create_train_test( train_size, test_size, number_max_len )

	def generate( self, data_size, number_max_len ):
		
		questions = []
		answers = []
		seen = set()
		
		f = lambda: ''.join( [ np.random.choice( list( '123456789' ) ) ] + \
								[ np.random.choice( list( '0123456789' ) ) \
									for i in range( np.random.randint( 1, number_max_len ) ) ] )
		g = lambda: np.random.choice( list( '+-*/' ) )
		h = lambda: np.random.choice( list( 'LR' ) )

		i = 0
		while i < data_size:
			n1, n2, n3 = f(), f(), f()
			o1, o2 = g(), g()
			
			q = []
			a = []
			
			if h() == 'L':
				if o1 in '*/':
					if o2 in '-+':
						q = [ n1, o1, '(', n2, o2, n3, ')', '=' ]
						a = [ n2, o2, n3, '=', o1, n1, '=' ]
					else:
						q = [ n1, o1, n2, o2, n3, '=' ]
						a = [ n1, o1, n2, o2, n3, '=' ]
				else:
					q = [ n1, o1, n2, o2, n3, '=' ]
					a = [ n1, o1, n2, o2, n3, '=' ]
			else:
				if o2 in '*/':
					if o2 in '-+':
						q = [ '(', n1, o1, n2, ')', o2, n3, '=' ]
						a = [ n1, o1, n2, '=', o2, n3, '=' ]
					else:
						q = [ n1, o1, n2, o2, n3, '=' ]
						a = [ n1, o1, n2, o2, n3, '=' ]
				else:
					q = [ n1, o1, n2, o2, n3, '=' ]
					a = [ n1, o1, n2, o2, n3, '=' ]
				
			qstr = ''.join( q )
			astr = ''.join( q )
			if qstr not in seen:
				questions.append( qstr )
				answers.append( astr )
				seen.add( qstr )
				i += 1

		return zip( questions, answers )
				
	def vectorize( self, data_set, sent_maxlen ):
		Q = []
		A = []

		for q, a in data_set:
			Q.append( self.alphabet.encode( q, sent_maxlen ) )
			A.append( self.alphabet.encode( a, sent_maxlen ) )

		return Q, A


	def create_train_test( self, train_size, test_size, number_max_len ):
		full_set = self.generate( train_size + test_size, number_max_len )
		self.train_set = full_set[ :train_size ]
		self.test_set = full_set[ train_size: ]

		# create alphabet
		# space means empty character and algo should ignore it
		self.alphabet = CharacterTable( ' 0123456789+-/*()=' )
		
		q_maxlen = max( map( len, (x for x, _ in full_set ) ) )
		a_maxlen = max( map( len, (x for _, x in full_set ) ) )
		self.sent_maxlen = max( q_maxlen, a_maxlen )
		
		self.train_q, self.train_a = self.vectorize( self.train_set, self.sent_maxlen )
		self.test_q, self.test_a = self.vectorize( self.test_set, self.sent_maxlen )

		print( 'Converting to numpy tensors' )
		def npize( x ):
			z = np.transpose( np.dstack( x ), [2, 0, 1] )
#			return np.reshape( z, [ z.shape[0], self.sent_maxlen * self.alphabet.maxlen ] )
			return z

		self.train_X = npize( self.train_q )
		self.train_y = npize( self.train_a )

	def decode( self, y ):
		pass

	def validate( self ):
		for q, a in zip( self.train_q, self.train_a ):
			qstr = datagen.alphabet.decode( q ).replace( '=', '' )
			astr = datagen.alphabet.decode( a ).replace( '=', '' )
			
			assert eval( qstr ) == eval( astr )

		for q, a in zip( self.test_q, self.test_a ):
			qstr = datagen.alphabet.decode( q ).replace( '=', '' )
			astr = datagen.alphabet.decode( a ).replace( '=', '' )
			
			assert eval( qstr ) == eval( astr )


class colors:
	ok = '\033[92m'
	fail = '\033[91m'
	close = '\033[0m'

# Parameters for the model and dataset
TRAINING_SIZE = 5000
DIGITS = 3
INVERT = True

# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
MAXLEN = DIGITS + 1 + DIGITS

print( 'Generating data...' )
datagen = DataGenerator( TRAINING_SIZE, TRAINING_SIZE / 5, MAXLEN )

print( 'Validating data' )
datagen.validate()

X, y = datagen.train_X, datagen.train_y

print( X.shape )
print( y.shape )

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
#indices = np.arange(len(y))
#np.random.shuffle(indices)
#X = X[indices]
#y = y[indices]

# Explicitly set apart 10% for validation data that we never train over
split_at = len(X) - len(X) / 10
X_train, X_val = slice_X( X, 0, split_at ), slice_X( X, split_at )
y_train, y_val = y[ : split_at ], y[ split_at : ]


print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add( RNN( HIDDEN_SIZE, input_shape=[ datagen.sent_maxlen, datagen.alphabet.maxlen ], return_sequences=True ) )

# The decoder RNN could be multiple layers stacked or a single layer
for _ in range( LAYERS ):
    model.add( RNN( HIDDEN_SIZE, return_sequences=True) )

# For each of step of the output sequence, decide which character should be chosen
model.add( TimeDistributed( Dense( len( datagen.alphabet.chars ) ) ) )
model.add( Activation( 'softmax' ) )

model.compile( loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[ 'accuracy' ] )

# Train the model each generation and show predictions against the validation dataset
for iteration in range( 1, 200 ):
    print()
    print( '-' * 50 )
    print( 'Iteration', iteration )
    model.fit( X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val) )
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint( 0, len( X_val ) )
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = datagen.alphabet.decode(rowX[0])
        correct = datagen.alphabet.decode(rowy[0])
        guess = datagen.alphabet.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if INVERT else q)
        print('T', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        print('---')
