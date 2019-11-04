"""
A simple library to create ODE based models.

Toy model makes it simple to create and reuse ODE models. Toy Model is both a
pedagogical tool, that makes it very easy to declare systems of ODEs, and is
also useful to  compose models from many sub-systems, which is typical from
many integrated assessment models.
"""
from .app import App
from .core import Model, Run, Value

__author__ = 'Fábio Macêdo Mendes'
__version__ = '0.1.0'
