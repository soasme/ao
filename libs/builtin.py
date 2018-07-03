MODULE = {}

def as_f(name):
    def deco(f):
        MODULE[name] = f
        return f
    return deco

@as_f('bool')
def builtin_bool(ctx):
    value = ctx.tos.pop()
    ctx.tos.push(_bool(value))

def _bool(value):
    return 'false' if value == 'false' or value == '[]' or value == '{}' \
            or value == 'null' or value == '""' or value == '0' \
            or value == '0.0' else 'true'

@as_f('assert')
def builtin_assert(ctx):
    value = ctx.tos.pop()
    if _bool(value) == 'false':
        ctx.machine.error = 'assertion error: %s' % value

def _type(value):
    if value.startswith('"') and value.endswith('"'):
        return '"string"'
    elif value.startswith('{') and value.endswith('}'):
        return '"object"'
    elif value.startswith('[') and value.endswith(']'):
        return '"array"'
    elif value.startswith('@'):
        return '"function"'
    elif value == 'true' or value == 'false':
        return '"bool"'
    elif value == 'null':
        return '"null"'
    else:
        return '"number"'

@as_f('type')
def builtin_type(ctx):
    ctx.tos.push(_type(ctx.tos.pop()))


def _add(v1, v2):
    if _type(v1) == '"number"' and _type(v2) == '"number"':
        if '.' in v1 or '.' in v2 or 'e' in v1 or 'E' in v1 or 'e' in v2 or 'E' in v2:
            return str(float(v1) + float(v2))
        else:
            return str(int(v1) + int(v2))
    elif _type(v1) == '"string"':
        if _type(v2) == '"string"':
            # abstract to str()
            return '"' + v1[1:max(1, len(v1)-1)] + v2[1:max(1,len(v2)-1)] + '"'
        elif _type(v2) == '"number"':
            return v1[0:max(1,len(v1)-1)] + str(v2) + '"'
    # support array and object merge, need type conversion
    else:
        raise ValueError

@as_f('add')
def builtin_add(ctx):
    right, left = ctx.tos.pop(), ctx.tos.pop()
    try:
        ctx.tos.push(_add(left, right))
    except ValueError:
        ctx.machine.error = 'type error: %s + %s' % (_type(left), _type(right))

@as_f('eq')
def builtin_eq(ctx):
    right, left = ctx.tos.pop(), ctx.tos.pop()
    ctx.tos.push(_eq(left, right))

def _eq(v1, v2):
    if _type(v1) == '"number"' and _type(v2) == '"number"':
        if '.' in v1 or '.' in v2 or 'e' in v1 or 'E' in v1 or 'e' in v2 or 'E' in v2:
            return str('true' if float(v1) == float(v2) else 'false')
        else:
            return str('true' if int(v1) == int(v2) else 'false')
    elif _type(v1) == '"string"' and _type(v2) == '"string"':
            return str('true' if v1 == v2 else 'false')
    elif _type(v1) == '"bool"' and _type(v2) == '"bool"':
        if v1 == v2: return 'true'
        else: return 'false'
    elif v1 == 'null' or v2 == 'null':
        if v1 == v2: return 'true'
        else: return 'false'
    # support array and object, need type caster and recursion
    else:
        raise ValueError
