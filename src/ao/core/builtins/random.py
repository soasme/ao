from rpython.rlib.rrandom import Random
from ao.core.types import Float

def random(params):
    return Float(Random().random())
