import sys

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

PRINT = 0
LOAD_LITERAL = 1
LOAD_VARIABLE = 2
SAVE_VARIABLE = 3

def get_location(pc, program):
    return "%s:%s" % (pc, program["instructions"][pc])


jitdriver = JitDriver(greens=['pc', 'program', ], reds=['machine'],
        get_printable_location=get_location)

def parse(source):
    return {
        "instructions": [
            "1 0",
            "0",
            "1 0",
            "3 0",
            "2 0",
            "0",
        ],
        "literals": [
            "hello world"
        ],
        "symbols": [
            "x"
        ]
    }

class Frame(object):
    def __init__(self, parent=None):
        self.parent = parent
        self.bindings = {}
    def save(self, key, value):
        self.bindings[key] = value
    def load(self, key):
        return self.bindings[key]


class Machine(object):

    def __init__(self):
        self.stack = []
        self.frame = Frame()
        self.exit_code = 0

    def run_code(self, program, pc):
        inst = [int(code) for code in program["instructions"][pc].split(' ')]
        opcode = inst[0]
        if opcode == PRINT:
            print(self.stack.pop())
        elif opcode == LOAD_LITERAL:
            self.stack.append(program["literals"][inst[1]])
        elif opcode == LOAD_VARIABLE:
            self.stack.append(self.frame.load(program["symbols"][inst[1]]))
        elif opcode == SAVE_VARIABLE:
            sym = program["symbols"][inst[1]]
            val = self.stack.pop()
            self.frame.save(sym, val)
        else:
            raise Exception("Unknown Bytecode")
        return pc + 1


def mainloop(program):
    pc = 0
    machine = Machine()
    while pc < len(program["instructions"]):
        jitdriver.jit_merge_point(pc=pc,
                                  program=program,
                                  machine=machine)
        pc = machine.run_code(program, pc)
    return machine.exit_code

def run(fp):
    program_contents = ""
    while True:
        read = os.read(fp, 4096)
        if len(read) == 0:
            break
        program_contents += read
    os.close(fp)
    program = parse(program_contents)
    return mainloop(program)

def entry_point(argv):
    try:
        filename = argv[1]
    except IndexError:
        print "You must supply a filename"
        return 1

    return run(os.open(filename, os.O_RDONLY, 0777))

def target(*args):
    return entry_point, None

def jitpolicy(driver):
    from rpython.jit.codewriter.policy import JitPolicy
    return JitPolicy()

if __name__ == "__main__":
    entry_point(sys.argv)
