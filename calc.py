from env import Env

class Calc( Env ):

	oper_2arg = { '-': lambda x,y: x - y,
					'+': lambda x,y: x + y,
					'*': lambda x,y: x * y,
					'/': lambda x,y: x / y }

	def __init__( self ):
		self.total_reward = 0
		self.reset()

	def reset( self, target=None ):
		self.acc = 0.0
		self.oper_acc = 0.0
		self.oper = '+'
		self.no_decimal = True
		self.no_oper = True
		self.decimal_location = 1

		self.target = target

	def result( self ):
		return self.acc
	
	def update( self, c ):
		
		if c in '0123456789':
			if self.no_decimal == True:
				self.acc = self.acc * 10.0 + float( c )
			else:
				self.acc = self.acc + float( c ) / (10.0 ** self.decimal_location)
				self.decimal_location += 1

		elif c == '.':
			self.no_decimal = False
			
		elif c in Calc.oper_2arg:
			if self.no_oper:
				self.oper_acc = self.acc
				self.acc = 0.0
				self.no_oper = False
				self.oper = c				
			else:
				self.acc = Calc.oper_2arg[ c ]( self.oper_acc, self.acc )
				self.oper_acc = self.acc
				self.no_decimal = True
				self.decimal_location = 1
				self.oper = c

		elif c == '=':
			self.acc = Calc.oper_2arg[ self.oper ]( self.oper_acc, self.acc )
			self.oper_acc = self.acc
			self.no_decimal = True
			self.decimal_location = 1
			self.oper = '+'
			self.no_oper = True
		
		elif c == 'C':
			self.acc = 0.0
			self.oper_acc = 0.0
			self.oper = '+'
			self.no_decimal = True
			self.decimal_location = 1
		
		else:
			raise ValueException( 'Invalid character: ' + c )

	def status( self ):
		r = self.result()
		check = r == self.target
		if check:
			self.total_reward += 1.0
		return str( r ), check, self.total_reward

if __name__ == '__main__':
	calc = Calc()
	
	while True:
		c = raw_input( '<: ' )
		calc.update( c )
		print( ':> {0}'.format( calc.result() ) )
