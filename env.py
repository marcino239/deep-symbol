from abc import ABCMeta, abstractmethod


class Env( object ):
	__metaclass__ = ABCMeta

	@abstractmethod
	def reset( self ):
		pass
	
	@abstractmethod
	def status( self ):
		pass

	@abstractmethod
	def update( self, c ):	
		pass
