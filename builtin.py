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


@as_f('system')
def builtin_system(ctx):
    assert len(ctx.params) == 1 and ctx.params[0].type == 'str'
    struct_name = ctx.params[0].strvalue
    lib = rdynload.dlopen(rffi.str2charp('libc.dylib'))
    func_ptr = rdynload.dlsym(lib, rffi.str2charp('system'))
    cif = lltype.malloc(jit_libffi.CIF_DESCRIPTION, 1, flavor='raw')
    cif.abi = clibffi.FFI_DEFAULT_ABI
    cif.nargs = 1
    cif.rtype = clibffi.ffi_type_sint32
    cif.atypes = lltype.malloc(clibffi.FFI_TYPE_PP.TO, 1, flavor='raw')
    cif.atypes[0] = clibffi.ffi_type_pointer
    cif.exchange_size = jit_libffi.SIZE_OF_FFI_ARG + jit_libffi.SIZE_OF_FFI_ARG
    cif.exchange_result = jit_libffi.SIZE_OF_FFI_ARG
    cif.exchange_args[0] = jit_libffi.SIZE_OF_FFI_ARG
    jit_libffi.jit_ffi_prep_cif(cif)
    exc = lltype.malloc(rffi.VOIDP.TO, cif.exchange_size, flavor='raw')
    argv0 = ctx.params[0].strvalue
    argv0_ptr = rffi.str2charp(argv0)
    argv0_ptr = rffi.cast(rffi.VOIDP, argv0_ptr)
    offset_0 = rffi.ptradd(exc, cif.exchange_args[0])
    exc_exchange_args0 = rffi.cast(rffi.VOIDPP, offset_0)
    exc_exchange_args0[0] = argv0_ptr
    jit_libffi.jit_ffi_call(cif, func_ptr, exc)
    result_ptr = rffi.ptradd(exc, cif.exchange_result)
    result_ptr = rffi.cast(rffi.SIGNEDP, result_ptr)
    result = result_ptr[0]
    lltype.free(exc, flavor='raw')
    # TODO: please provide a `newrawint` function (int)->Int.
    ctx.tos.push(ctx.machine.space.newint(str(result)))
