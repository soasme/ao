from rpython.rtyper.lltypesystem.lltype import *
from rpython.rlib import rdynload, objectmodel, clibffi, rgc, rgil, jit_libffi
from rpython.rtyper.lltypesystem import rffi, lltype, llmemory
import rpython.rlib.jit as jit

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
        ctx.machine.error = ctx.machine.space.newstr('assertion error')

@as_f('type')
def builtin_type(ctx):
    assert len(ctx.params) == 1
    val = ctx.params[0].space.newstr(ctx.params[0].type)
    ctx.tos.push(val)

@as_f('raise')
def builtin_raise(ctx):
    assert len(ctx.params) == 1
    ctx.machine.error = ctx.params[0]


@as_f('__ffi_dynload__')
def builtin_ffi_dynload(ctx):
    assert len(ctx.params) == 2 and ctx.params[0].type == 'str' and ctx.params[0].type == 'str'
    libname = ctx.params[0].strvalue
    libpath = ctx.params[1].strvalue

    # load dylib if haven't
    if libname in ctx.machine.space.ffi_libs:
        ctx.tos.push(ctx.machine.space.null)
        return

    # try to load dylib
    path = rffi.str2charp(libpath)
    try:
        lib = rdynload.dlopen(path)
    except rdynload.DLOpenError as e:
        ctx.machine.error = ctx.machine.space.newstr('unable to load library: %s' % libname)
        return
    finally:
        lltype.free(path, flavor='raw')

    # cache dylib
    ctx.machine.space.ffi_libs[libname] = ctx.machine.space.newforeignlibrary(libname, lib)
    ctx.tos.push(ctx.machine.space.null)


CFFI_TYPES = {
    'int': clibffi.ffi_type_sint64,
    'long': clibffi.ffi_type_sint64,
    'double': clibffi.ffi_type_double,
}

def _cast_aotype_to_ffitype(rtype):
    if rtype.endswith('*'):
        return clibffi.ffi_type_pointer
    return CFFI_TYPES[rtype]

def _cast_aovalue_to_ffivalue(space, value, type, ptr):
    if value.type == 'str' and type == 'char *':
        pnt = rffi.cast(rffi.CCHARPP, ptr)
        pnt[0] = rffi.str2charp(value.strvalue)
    elif value.type == 'int' and type == 'int':
        pnt = rffi.cast(rffi.INTP, ptr)
        pnt[0] = rffi.cast(rffi.INT, value.intvalue)
    elif value.type == 'float' and type == 'float':
        pnt = rffi.cast(rffi.FLOATP, ptr)
        pnt[0] = rffi.cast(rffi.FLOAT, value.floatvalue)
    elif value.type == 'float' and type == 'double':
        pnt = rffi.cast(rffi.DOUBLEP, ptr)
        pnt[0] = rffi.cast(rffi.DOUBLE, value.floatvalue)
    else:
        raise ValueError('not implemented aovalue->ffivalue.')

def _cast_ffivalue_to_aovalue(space, ptr, type):
    if type == 'int':
        ptn = rffi.cast(rffi.INTP, ptr)
        return space.newrawint(rffi.cast(rffi.LONG, ptn[0]))
    elif type == 'double':
        ptn = rffi.cast(rffi.DOUBLEP, ptr)
        return space.newrawfloat(rffi.cast(rffi.DOUBLE, ptn[0]))
    else:
        raise ValueError('not implemented.')

def _align(n): return (n + 7) & ~7

@as_f('__ffi_function__')
def builtin_ffi_function(ctx):
    assert len(ctx.params) == 4 and \
            ctx.params[0].type == 'str' and \
            ctx.params[1].type == 'str' \
            and ctx.params[2].type == 'str' \
            and ctx.params[3].type == 'array'
    for e in ctx.params[3].arrayvalue:
        assert e.type == 'str'
    libname = ctx.params[0].strvalue
    funcname = ctx.params[1].strvalue
    rtype = ctx.params[2].strvalue
    atypes = [e.strvalue for e in ctx.params[3].arrayvalue]

    argc = len(atypes)
    cif = lltype.malloc(jit_libffi.CIF_DESCRIPTION, argc, flavor='raw')
    cif.abi = clibffi.FFI_DEFAULT_ABI
    cif.nargs = argc
    cif.rtype = _cast_aotype_to_ffitype(rtype)
    cif.atypes = lltype.malloc(clibffi.FFI_TYPE_PP.TO, argc, flavor='raw')

    # create room for an array of nargs pointers
    exchange_offset = rffi.sizeof(rffi.VOIDP) * argc
    exchange_offset = _align(exchange_offset)
    cif.exchange_result = exchange_offset

    # create room for return value, roundup to sizeof(ffi-arg)
    exchange_offset += max(rffi.getintfield(cif.rtype, 'c_size'), jit_libffi.SIZE_OF_FFI_ARG)

    # set size for each arg
    for i in range(argc):
        atype = _cast_aotype_to_ffitype(atypes[i])
        cif.atypes[i] = atype
        exchange_offset = _align(exchange_offset)
        cif.exchange_args[i] = exchange_offset
        exchange_offset += rffi.getintfield(atype, 'c_size')

    cif.exchange_size = exchange_offset
    code = jit_libffi.jit_ffi_prep_cif(cif)
    if code != clibffi.FFI_OK:
        ctx.machine.error = ctx.machine.space.newstr('failed to build function %s for lib %s.' % (funcname, libname))
    ctx.machine.space.ffi_functions[(libname, funcname)] = ctx.machine.space.newforeignfunction(
            libname, funcname, rtype, atypes, cif)
    ctx.tos.push(ctx.machine.space.null)


@as_f('__ffi_struct__')
def builtin_ffi_struct(ctx):
    pass

@as_f('__ffi_union__')
def builtin_ffi_struct(ctx):
    pass

@as_f('__ffi_enum__')
def builtin_ffi_enum(ctx):
    pass

@as_f('__ffi_typedef__')
def builtin_ffi_typedef(ctx):
    pass

@as_f('__ffi_call__')
def builtin_ffi_call(ctx):
    assert len(ctx.params) == 3 and \
            ctx.params[0].type == 'str' and \
            ctx.params[1].type == 'str' and \
            ctx.params[2].type == 'array'

    libname = ctx.params[0].strvalue
    symname = ctx.params[1].strvalue
    args = ctx.params[2].arrayvalue
    lib = ctx.machine.space.ffi_libs[libname].lib
    ff = ctx.machine.space.ffi_functions[(libname, symname)]
    cif = ff.cif
    func = rdynload.dlsym(lib, rffi.str2charp(symname))
    exc = lltype.malloc(rffi.VOIDP.TO, cif.exchange_size, flavor='raw')
    ptr = exc
    for i in range(cif.nargs):
        ptr = rffi.ptradd(exc, cif.exchange_args[i])
        _cast_aovalue_to_ffivalue(ctx.machine.space, args[i], ff.atypes[i], ptr)

    jit_libffi.jit_ffi_call(cif, func, exc)

    ptr = rffi.ptradd(exc, cif.exchange_result)
    val = _cast_ffivalue_to_aovalue(ctx.machine.space, ptr, ff.rtype)
    lltype.free(exc, flavor='raw')
    ctx.tos.push(val)
