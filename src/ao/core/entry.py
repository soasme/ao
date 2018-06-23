import os
import sys

try:
    from rpython.rlib.jit import JitDriver, purefunction
except ImportError:
    class JitDriver(object):
        def __init__(self,**kw): pass
        def jit_merge_point(self,**kw): pass
        def can_enter_jit(self,**kw): pass
    def purefunction(f): return f

from ao.core.machine import Machine, parse

def get_location(pc, program, context):
    return "%s_%s" % (pc, program[pc])


jitdriver = JitDriver(greens=['pc', 'program', 'context', ], reds=['machine'],
        get_printable_location=get_location)

def mainloop(program, context):
    pc = 0
    machine = Machine()
    while pc < len(program):
        jitdriver.jit_merge_point(pc=pc,
                                  program=program,
                                  context=context,
                                  machine=machine)
        pc = machine.run_code(program, pc, context)

def run(fp):
    program_contents = ""
    while True:
        read = os.read(fp, 4096)
        if len(read) == 0:
            break
        program_contents += read
    os.close(fp)
    program, context = parse(program_contents)
    mainloop(program, context)

def run_script(filename):
    run(os.open(filename, os.O_RDONLY, 0777))
    return 0
