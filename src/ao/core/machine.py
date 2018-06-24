from ao.core.builtins.registry import BUILTIN_REGISTRY
from ao.core.types import Int, Str, Float, Bool, Null, Array, Object, Env, Builtin

LOAD_LITERAL = 1
LOAD_VARIABLE = 2
LOAD_BUILTIN = 3
LOAD_CLOSURE = 4
PRINT = 5
ARITH_OP = 6
STORE_VARIABLE = 9
MAKE_FUNCTION = 10
CALL_FUNCTION = 11
RETURN_VALUE = 12
MAKE_OBJECT = 13
MAKE_ARRAY = 14

ARITH_ADD = 1
ARITH_SUB = 2
ARITH_MUL = 3
ARITH_DIV = 4
ARITH_MOD = 5
ARITH_LEFTSHIFT = 6
ARITH_RIGHTSHIFT = 7
ARITH_BITAND = 8
ARITH_BITOR = 9

LOGIC_AND = 1
LOGIC_OR = 2
LOGIC_NOT = 3

def op_add(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.value() + right.value())
    else:
        raise Exception('unknown type for add: %s, %s' % ('Type1', 'Type2'))

def op_sub(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.value() - right.value())
    else:
        raise Exception('unknown type for sub: %s, %s' % ('Type1', 'Type2'))

def op_mul(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.value() * right.value())
    else:
        raise Exception('unknown type for mul: %s, %s' % ('Type1', 'Type2'))

def op_div(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.value() / right.value())
    else:
        raise Exception('unknown type for div: %s, %s' % ('Type1', 'Type2'))

def op_mod(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.value() % right.value())
    else:
        raise Exception('unknown type for mod: %s, %s' % ('Type1', 'Type2'))

def op_leftshift(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.value() << right.value())
    else:
        raise Exception('unknown type for mod: %s, %s' % ('Type1', 'Type2'))

def op_rightshift(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.value() >> right.value())
    else:
        raise Exception('unknown type for mod: %s, %s' % ('Type1', 'Type2'))

def op_bitand(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.value() & right.value())
    else:
        raise Exception('unknown type for mod: %s, %s' % ('Type1', 'Type2'))

def op_bitor(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.value() | right.value())
    else:
        raise Exception('unknown type for mod: %s, %s' % ('Type1', 'Type2'))

def op_and(tos):
    return isinstance(tos, Bool) and tos.value()

def call_function(f, params):
    if isinstance(f, Builtin):
        return BUILTIN_REGISTRY[f.code](params)
    raise Exception('too many arguments')

class Machine(object):

    def __init__(self):
        self.env = Env(parent=None)
        self.stack = [[]]

    def run_code(self, program, pc, context):
        code = program[pc]
        frame = self.stack[0]
        if code[0] == LOAD_LITERAL:
            frame.append(context["literals"][code[1]])
        elif code[0] == LOAD_VARIABLE:
            key = context["symbols"][code[1]]
            frame.append(self.env.resolve(key))
        elif code[0] == STORE_VARIABLE:
            key = context["symbols"][code[1]]
            tos = frame.pop()
            self.env.store(key, tos)
        elif code[0] == LOAD_BUILTIN:
            frame.append(Builtin(code[1]))
        elif code[0] == CALL_FUNCTION:
            parameters = []
            for _ in range(code[1]):
                parameters.append(frame.pop())
            f = frame.pop()
            r = call_function(f, parameters)
            frame.append(r)
        elif code[0] == ARITH_OP:
            if code[1] == ARITH_ADD:
                frame.append(op_add(frame.pop(), frame.pop()))
            elif code[1] == ARITH_SUB:
                frame.append(op_sub(frame.pop(), frame.pop()))
            elif code[1] == ARITH_MUL:
                frame.append(op_mul(frame.pop(), frame.pop()))
            elif code[1] == ARITH_DIV:
                frame.append(op_div(frame.pop(), frame.pop()))
            elif code[1] == ARITH_MOD:
                frame.append(op_mod(frame.pop(), frame.pop()))
            elif code[1] == ARITH_LEFTSHIFT:
                frame.append(op_leftshift(frame.pop(), frame.pop()))
            elif code[1] == ARITH_RIGHTSHIFT:
                frame.append(op_rightshift(frame.pop(), frame.pop()))
            elif code[1] == ARITH_BITAND:
                frame.append(op_bitand(frame.pop(), frame.pop()))
            elif code[1] == ARITH_BITOR:
                frame.append(op_bitor(frame.pop(), frame.pop()))
            else:
                raise Exception('unknown operator %d', code[1])
        elif code[0] == PRINT:
            print(frame.pop().str())
        return pc + 1


def parse(content):
    # TODO: need to parse content and content dependencies.
    return [
        [LOAD_LITERAL, 0],
        [LOAD_LITERAL, 1],
        [ARITH_OP, ARITH_ADD],
        [PRINT],
        [LOAD_LITERAL, 2],
        [STORE_VARIABLE, 0],
        [LOAD_VARIABLE, 0],
        [PRINT],
        [LOAD_BUILTIN, 1],
        [CALL_FUNCTION, 0],
        [PRINT],
    ], {
        "literals": [Int(1), Int(2), Str("hello world"), ],
        "symbols": [Str("abc")],
    }
