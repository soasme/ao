LOAD_LITERAL = 1
LOAD_VARIABLE = 2
LOAD_BUILTIN = 3
LOAD_CLOSURE = 4
PRINT = 5
ARITH_OP = 6
LOGIC_OP = 7
STORE_VARIABLE = 8
MAKE_FUNCTION = 9
CALL_FUNCTION = 10
RETURN_VALUE = 11
MAKE_OBJECT = 12
MAKE_ARRAY = 13

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

# def op_and(left, right):
    # if left is true and right is true:
        # return true
    # else:
        # return false

# def op_or(left, right):
    # if isinstance(left, Bool) and isinstance(right, Bool):
        # return Bool(left.value() or right.value())
    # else:
        # raise Exception('unknown type for or: %s, %s' % ('Type1', 'Type2'))

# def op_not(tos):
    # if isinstance(tos, Bool):
        # return Bool(not tos.value())
    # else:
        # raise Exception('unknown type for not: %s' % ('Type'))

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
        # elif code[0] == LOGIC_OP:
            # if code[1] == LOGIC_AND:
                # frame.append(op_and(frame.pop(), frame.pop()))
            # elif code[1] == LOGIC_OR:
                # frame.append(op_or(frame.pop(), frame.pop()))
            # elif code[1] == LOGIC_NOT:
                # frame.append(op_not(frame.pop()))
            # else:
                # raise Exception('unknown operator %d', code[1])
        elif code[0] == PRINT:
            print(frame.pop().str())
        return pc + 1


class Literal(object):
    pass
class Int(Literal):
    def __init__(self, val):
        self.intval = val
    def value(self):
        return self.intval
    def str(self):
        return str(self.intval)
class Str(Literal):
    def __init__(self, val):
        self.strval = val
    def value(self):
        return self.strval
    def str(self):
        return str(self.strval)

class Bool(Literal):
    def __init__(self, val):
        self.boolval = val
    def value(self):
        return self.boolval
    def str(self):
        return str(self.boolval)
class Null(Literal):
    def __init__(self, val):
        self.nullval = val
    def value(self):
        return self.nullval
    def str(self):
        return "null"
class Float(Literal):
    def __init__(self, val):
        self.floatval = val
    def value(self):
        return self.floatval
    def str(self):
        return str(self.floatval)
class Array(Literal):
    def __init__(self, val):
        self.arrayval = val
    def value(self):
        return self.arrayval
    def str(self):
        return "[%s]" % (", ".join([e.str() for e in self.arrayval]))
class Object(Literal):
    def __init__(self, val):
        self.objectval = val
    def value(self):
        return self.objectval
    def str(self):
        return "{object}"
class Env(Object):

    def __init__(self, parent):
        self.parent = parent
        self.bindings = {}

    def resolve(self, key):
        if not isinstance(key, Str):
            raise Exception('unknown variable name `%s`.' % key.str())
        _key = key.value()
        if _key not in self.bindings:
            if self.parent is None:
                raise Exception('undefined variable `%s`.' % key)
            return self.parent.resolve(_key)
        return self.bindings[_key]

    def store(self, key, value):
        if not isinstance(key, Str):
            raise Exception('unknown variable name `%s`.' % key.str())
        self.bindings[key.value()] = value


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
    ], {
        "literals": [Int(1), Int(2), Str("hello world")],
        "symbols": [Str("abc")],
    }
