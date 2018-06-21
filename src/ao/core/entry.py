import os
import sys

from ao.core.machine import Machine, parse, jitdriver

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
