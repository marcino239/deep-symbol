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

f1 = plt.figure()
f2 = plt.figure()
ax1 = f1.add_subplot(111)
ax2 = f2.add_subplot(111)

test_hidden = []
for f in flist:
	if f.startswith( 'test_hidden_size_layers_' ):
		terms = re.split( r'[_\.]+', f )
		h, l = map( float, [ terms[ 4 ], terms[ 5 ] ] )
		
		dat = pickle.load( open( f, 'r' ) )
		ax1.plot( dat[ 'val_loss' ], label='loss_{0}_{1}'.format( h, l ), marker=markers.next() )
		ax2.plot( dat[ 'val_acc' ], label='acc_{0}_{1}'.format( h, l ), marker=markers.next() )

ax1.legend( loc=1 )
ax1.set_xlabel( 'Epochs' )
ax1.set_ylabel( 'Loss' )

ax2.legend( loc=4 )
ax2.set_xlabel( 'Epochs' )
ax2.set_ylabel( 'Accuracy' )

plt.show()
