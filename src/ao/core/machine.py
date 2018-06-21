try:
    from rpython.rlib.jit import JitDriver, purefunction
except ImportError:
    class JitDriver(object):
        def __init__(self,**kw): pass
        def jit_merge_point(self,**kw): pass
        def can_enter_jit(self,**kw): pass
    def purefunction(f): return f

def get_location(pc, program, context):
    return "%s_%s" % (pc, program[pc])

jitdriver = JitDriver(greens=['pc', 'program', 'context', ], reds=['machine'],
        get_printable_location=get_location)

LOAD_LITERAL = 1
LOAD_VARIABLE = 2
LOAD_BUILTIN = 3
LOAD_CLOSURE = 4
PRINT = 5
ARITH_ADD = 6

def op_add(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.value() + right.value())
    else:
        raise Exception('unknown type for +: %s, %s' % ('Type1', 'Type2'))

class Env(object):

    def __init__(self, parent):
        self.parent = parent
        self.bindings = {}

    def resolve(self, key):
        if self.parent is None:
            raise Exception('undefined variable `%s`.' % key)
        if key not in self.bindings:
            return self.parent.resolve(key)
        return self.bindings[key]

class Machine(object):

    def __init__(self):
        self.env = Env(parent=None)
        self.stacks = {"main": []}
        self.stack = "main"

    def run_code(self, program, pc, context):
        code = program[pc]
        stack = self.stacks[self.stack]
        if code[0] == LOAD_LITERAL:
            val = context["literals"][code[1]]
            stack.append(val)
        elif code[0] == ARITH_ADD:
            stack.append(op_add(stack.pop(), stack.pop()))
        elif code[0] == PRINT:
            print(stack.pop().str())
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

def parse(content):
    # TODO: need to parse content and content dependencies.
    return [
        [LOAD_LITERAL, 0],
        [LOAD_LITERAL, 1],
        [ARITH_ADD],
        [PRINT],
        [LOAD_LITERAL, 2],
        [PRINT],
    ], {
        "literals": [Int(1), Int(2), Str("hello world")]
    }
