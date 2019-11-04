import operator as op

from lark import Lark, InlineTransformer
from sympy import S
from sympy.physics import units
from sympy.physics.units import Quantity
from sympy.physics.units.dimensions import dimsys_SI

from sidekick import namespace

DIMENSIONLESS = Quantity("dimensionless")
DIMENSIONLESS.set_dimension(S.One)
DIMENSIONLESS.set_scale_factor(S.One)
SIMPY_UNITS = {k: v for k, v in vars(units).items() if not k.startswith('_')}
UNITS = namespace(
    **SIMPY_UNITS,
)

grammar = Lark(r"""
?start : expr
       | NUMBER           -> dimensionless
        
?expr  : expr "*" atom    -> mul
       | expr atom        -> mul
       | expr "/" atom    -> div
       | atom
     
?atom  : name "^" number  -> pow
       | name number      -> pow
       | name
       
name   : NAME 
number : NUMBER 
     
NUMBER : /-?\d+/ 
NAME   : /[a-zA-Z$%]+/

%ignore /\s+/
""", parser='lalr')


class UnitTransformer(InlineTransformer):
    number = int
    pow = op.pow
    mul = op.mul
    div = op.truediv

    def __init__(self, system=None):
        self.system = system or dimsys_SI
        super().__init__()

    def name(self, name):
        return UNITS[str(name)]

    def dimensionless(self, N):
        return int(N) * DIMENSIONLESS


def parse_unit(src, system=None):
    """
    Parse string describing unit.
    """
    tree = grammar.parse(src)
    transformer = UnitTransformer(system)
    return transformer.transform(tree)


def parse_unit_msg(u):
    """
    Parse string of the form "[unit] message"
    """

    pre, _, msg = u.partition(']')
    return parse_unit(pre[1:]), msg.strip()

