from calc import Calc

def test_input():
	c = Calc()
	
	c.update( '1' )
	c.update( '2' )
	c.update( '3' )
	s = str( c.result() )
	assert s == '123.0'
	
def test_input2():
	c = Calc()
	
	c.update( '1' )
	c.update( '2' )
	c.update( '3' )
	c.update( '*' )
	c.update( '3' )
	c.update( '2' )
	c.update( '1' )

	s = str( c.result() )
	assert s == str( 321.0 )

def test_input3():
	c = Calc()
	
	c.update( '1' )
	c.update( '2' )
	c.update( '3' )
	c.update( '*' )
	c.update( '3' )
	c.update( '2' )
	c.update( '1' )
	c.update( '=' )
	c.update( 'C' )

	s = str( c.result() )
	assert s == str( 0.0 )

def test_mul():
	c = Calc()
	
	c.update( '1' )
	c.update( '2' )
	c.update( '3' )
	c.update( '*' )
	c.update( '3' )
	c.update( '2' )
	c.update( '1' )
	c.update( '=' )

	s = str( c.result() )
	assert s == str( 123 * 321.0 )
