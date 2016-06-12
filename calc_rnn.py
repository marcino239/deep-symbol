# -*- coding: utf-8 -*-
'''Teaches network to solve a brackets priority problem
adapted from keras examples: 
https://raw.githubusercontent.com/fchollet/keras/master/examples/addition_rnn.py
'''

from __future__ import print_function

import numpy as np
import keras
import tensorflow as tf
import sys
import cPickle as pickle

from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Dropout, Merge
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
	def __init__( self, train_size, test_size, number_max_len, dataset_type='calc' ):
		self.dataset_type = dataset_type
		self.create_train_test( train_size, test_size, number_max_len )

	def get_q_a( self, n, o, bracket ):
		n1, n2, n3 = n
		o1, o2 = o

		if self.dataset_type == 'calc':
			if bracket == 'L':
				q = [ n1, o1, '(', n2, o2, n3, ')', '=' ]
				a = [ n2, o2, n3, '=', o1, n1, '=' ]
			else:
				q = [ '(', n1, o1, n2, ')', o2, n3, '=' ]
				a = [ n1, o1, n2, '=', o2, n3, '=' ]

		elif self.dataset_type == 'calc_old':
			if bracket == 'L':
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

		elif self.dataset_type == 'reverse_seq':
			q = [ n1, '+', n2, '+', n3, '=' ]
			a = [ n3, '+', n2, '+', n1, '=' ]

		return q, a

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
			n = [ f(), f(), f() ]
			o = [  g(), g() ]
			
			q, a = self.get_q_a( n, o, h() )
			
			qstr = ''.join( q )
			astr = ''.join( a )
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
			try:
				qstr = datagen.alphabet.decode( q ).replace( '=', '' )
				astr = datagen.alphabet.decode( a ).replace( '=', '' )
			
				assert eval( qstr ) == eval( astr )
			except ZeroDivisionError as e:
				print( 'q: {0}, a: {1}'.format( qstr, astr ) )

		for q, a in zip( self.test_q, self.test_a ):
			try:
				qstr = datagen.alphabet.decode( q ).replace( '=', '' )
				astr = datagen.alphabet.decode( a ).replace( '=', '' )
			
				assert eval( qstr ) == eval( astr )

			except ZeroDivisionError as e:
				print( 'q: {0}, a: {1}'.format( qstr, astr ) )


class colors:
	ok = '\033[92m'
	fail = '\033[91m'
	close = '\033[0m'

# Parameters for the model and dataset
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer( 'TRAINING_SIZE', 100000, 'size of training set' )
flags.DEFINE_integer( 'DIGITS', 3, 'max number of digits in the number' )
flags.DEFINE_boolean( 'INVERT', False, 'invert output sequence' )

flags.DEFINE_integer( 'HIDDEN_SIZE', 50, 'size of hidden layer' )
flags.DEFINE_integer( 'BATCH_SIZE', 512, 'batch size' )
flags.DEFINE_integer( 'LAYERS', 2, 'nuber of hidden layers' )
flags.DEFINE_integer( 'EPOCHS', 200, 'nuber of epochs' )

flags.DEFINE_boolean( 'BIDIRECTIONAL', True, 'use bidirectional mode' )
flags.DEFINE_string( 'CELL_TYPE', 'LSTM', 'cell type: LSTM, GRU, SimpleRNN' )

flags.DEFINE_string( 'DATASET_TYPE', 'reverse_seq', 'dataset type: reverse_seq, calc, calc_old' )
flags.DEFINE_string( 'MODEL_NAME', '', 'model file name' )
flags.DEFINE_string( 'ACC_LOSS_FNAME', '', 'accuracy and loss file name' )

flags.DEFINE_float( 'DROPOUT', 0.3, 'dropout probability' )

if FLAGS.CELL_TYPE == 'LSTM':
	RNN = recurrent.LSTM
elif FLAGS.CELL_TYPE == 'GRU':
	RNN = recurrent.LSTM
elif FLAGS.CELL_TYPE == 'SimpleRNN':
	RNN = recurrent.LSTM
else:
	raise ValueError( 'invalid CELL_TYPE: ' + FLAGS.CELL_TYPE )

print( 'Generating data' )
datagen = DataGenerator( FLAGS.TRAINING_SIZE, 0, FLAGS.DIGITS, FLAGS.DATASET_TYPE )

#print( 'Validating data' )
#datagen.validate()

X, y = datagen.train_X, datagen.train_y

print( X.shape )
print( y.shape )

# Explicitly set apart 10% for validation data that we never train over
split_at = len(X) - len(X) / 10
X_train, X_val = slice_X( X, 0, split_at ), slice_X( X, split_at )
y_train, y_val = y[ : split_at ], y[ split_at : ]


print('Build model')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).

left = Sequential()
left.add( RNN( FLAGS.HIDDEN_SIZE, input_shape=[ datagen.sent_maxlen, datagen.alphabet.maxlen ], return_sequences=True ) )
if FLAGS.BIDIRECTIONAL:
	right = Sequential()
	right.add( RNN( FLAGS.HIDDEN_SIZE, input_shape=[ datagen.sent_maxlen, datagen.alphabet.maxlen ],
			 return_sequences=True, go_backwards = True ) )
	
	model.add( Merge( [ left, right ], mode = 'sum' ) )
else:
	model.add( left, mode = 'sum' )

if FLAGS.DROPOUT > 0.0:
	model.add( Dropout( FLAGS.DROPOUT ) )

def fork_model( model ):
	forks = []
	for i in range( 2 ):
		f = Sequential()
		f.add( model )
		forks.append( f )

	return forks

# The decoder RNN could be multiple layers stacked or a single layer
for _ in range( FLAGS.LAYERS ):
	if FLAGS.BIDIRECTIONAL:
		left, right = fork_model( model )
		left.add( RNN( FLAGS.HIDDEN_SIZE, return_sequences=True ) )
		right.add( RNN( FLAGS.HIDDEN_SIZE, return_sequences=True, go_backwards=True ) )
	else:
		model.add( RNN( FLAGS.HIDDEN_SIZE, return_sequences=True ) )

	model.add( Dropout( FLAGS.DROPOUT ) )


# For each of step of the output sequence, decide which character should be chosen
model.add( Dropout( FLAGS.DROPOUT ) )
model.add( TimeDistributed( Dense( len( datagen.alphabet.chars ) ) ) )
model.add( Activation( 'softmax' ) )

model.compile( loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[ 'accuracy' ] )

# stream events
keras.callbacks.RemoteMonitor( root='http://localhost:9000' )

# Train the model each generation and show predictions against the validation dataset
print( '-' * 20 + 'TRAINING' + '-' * 20 )

x_train_in = [ X_train, X_train ] if FLAGS.BIDIRECTIONAL else X_train
x_val_in = [ X_val, X_val ] if FLAGS.BIDIRECTIONAL else X_val

hist = model.fit( x_train_in, y_train, batch_size=FLAGS.BATCH_SIZE,
					nb_epoch=FLAGS.EPOCHS,
					shuffle=True,
					validation_data=[ x_val_in, y_val ] )

if FLAGS.ACC_LOSS_FNAME != '':
	d = {}
	d[ 'val_loss' ] = hist.history[ 'val_loss' ]
	d[ 'val_acc' ] = hist.history[ 'val_acc' ]
	accu_loss_file = pickle.dump( d, open( FLAGS.ACC_LOSS_FNAME, 'wb' ) )

# Select 20 samples from the validation set at random so we can visualize errors
for i in range( 20 ):
	ind = np.random.randint( 0, len( X_val ) )
	rowX, rowy = X_val[ np.array( [ind] ) ], y_val[ np.array( [ind] ) ]
	q = datagen.alphabet.decode( rowX[ 0 ] )
	rowX = [ rowX, rowX ] if FLAGS.BIDIRECTIONAL else rowX

	preds = model.predict_classes(rowX, verbose=0)

	correct = datagen.alphabet.decode(rowy[0])
	guess = datagen.alphabet.decode(preds[0], calc_argmax=False)
	print( 'Q', q[::-1] if FLAGS.INVERT else q )
	print( 'A', correct)
	print( colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess )
	print( '---' )

print( 'last loss: {0}, acc: {1}'.format( hist.history[ 'val_loss' ][-1], hist.history[ 'val_acc' ][-1] ) )

# save model
if FLAGS.MODEL_NAME != '':
	open( FLAGS.MODEL_NAME + '.json', 'w' ).write( model.to_json() )
	model.save_weights( FLAGS.MODEL_NAME + '_weights.h5' )
