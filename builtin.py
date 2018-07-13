MODULE = {}

def as_f(name):
    def deco(f):
        MODULE[name] = f
        return f
    return deco

@as_f('bool')
def builtin_bool(ctx):
    assert len(ctx.params) == 1
    pyval = ctx.params[0].space.castbool(ctx.params[0])
    if pyval:
        ctx.tos.push(ctx.params[0].space.true)
    else:
        ctx.tos.push(ctx.params[0].space.false)

@as_f('assert')
def builtin_assert(ctx):
    assert len(ctx.params) == 1
    pyval = ctx.params[0].space.castbool(ctx.params[0])
    if not pyval:
        ctx.machine.error = 'assertion error'

@as_f('type')
def builtin_type(ctx):
    assert len(ctx.params) == 1
    val = ctx.params[0].space.newstr(ctx.params[0].type)
    ctx.tos.push(val)
