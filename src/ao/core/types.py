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
class Builtin(Literal):
    def __init__(self, code):
        self.code = code

