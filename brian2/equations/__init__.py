'''
Module handling equations and "code strings", expressions or statements, used
for example for the reset and threshold definition of a neuron.
'''
from .codestrings import Expression, ExpressionTemplate, Statements
from .equations import Equations

__all__ = ['Equations', 'Expression',
           'ExpressionTemplate', 'Statements']
