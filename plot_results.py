import glob
import matplotlib.pyplot as plt
import re
import cPickle as pickle
import itertools

flist = glob.glob( '*.res' )
flist.sort()

print( 'processing:' )
for f in flist:
	print( f )

markers = itertools.cycle( [ 'x', '+', '^', 'o', '*' ] )

f = {}
a = {}

f[ 'hsl1' ] = plt.figure()
f[ 'hsl2' ] = plt.figure()

f[ 'd1' ] = plt.figure()
f[ 'd2' ] = plt.figure()

a[ 'hsl_ax1' ] = f[ 'hsl1' ].add_subplot( 111 )
a[ 'hsl_ax2' ] = f[ 'hsl2' ].add_subplot( 111 )

a[ 'd_ax1' ] = f[ 'd1' ].add_subplot( 111 )
a[ 'd_ax2' ] = f[ 'd2' ].add_subplot( 111 )

test_hidden = []
for f in flist:
	if f.startswith( 'test_hidden_size_layers_' ):
		terms = re.split( r'[_\.]+', f )
		h, l = map( float, [ terms[ 4 ], terms[ 5 ] ] )
		
		dat = pickle.load( open( f, 'r' ) )
		a[ 'hsl_ax1' ].plot( dat[ 'val_loss' ], label='hl_loss_{0}_{1}'.format( h, l ), marker=markers.next() )
		a[ 'hsl_ax2' ].plot( dat[ 'val_acc' ], label='hl_acc_{0}_{1}'.format( h, l ), marker=markers.next() )
		
	elif f.startswith( 'test_dropout_' ):
		terms = re.split( r'[_\.]+', f )
		d = float( terms[3] ) / 10.0
		
		dat = pickle.load( open( f, 'r' ) )
		a[ 'd_ax1' ].plot( dat[ 'val_loss' ], label='d_loss_{0}'.format( d ), marker=markers.next() )
		a[ 'd_ax2' ].plot( dat[ 'val_acc' ], label='d_acc_{0}'.format( d ), marker=markers.next() )


a[ 'hsl_ax1' ].legend( loc=1 )
a[ 'hsl_ax1' ].set_xlabel( 'Epochs' )
a[ 'hsl_ax1' ].set_ylabel( 'Loss' )

a[ 'hsl_ax2' ].legend( loc=4 )
a[ 'hsl_ax2' ].set_xlabel( 'Epochs' )
a[ 'hsl_ax2' ].set_ylabel( 'Loss' )

a[ 'd_ax1' ].legend( loc=1 )
a[ 'd_ax1' ].set_xlabel( 'Epochs' )
a[ 'd_ax1' ].set_ylabel( 'Loss' )

a[ 'd_ax2' ].legend( loc=4 )
a[ 'd_ax2' ].set_xlabel( 'Epochs' )
a[ 'd_ax2' ].set_ylabel( 'Loss' )


plt.show()
