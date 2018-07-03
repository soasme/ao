MODULE = {}

def as_f(name):
    def deco(f):
        MODULE[name] = f
        return f
    return deco

@as_f('bool')
def bool(ctx):
    value = ctx.tos.pop()
    bvalue = 'true' if value == 'false' or value == '[]' or value == '{}' \
                or value == '""' or value == '0' or value == '0.0' else 'false'
    ctx.tos.push(bvalue)

# @as_f('assert')
# def assert(ctx):
    # value = ctx.tos.pop()
    # if value == 'value':
        # ctx.tos.
