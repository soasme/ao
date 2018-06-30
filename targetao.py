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
CALL_FUNCTION = 4
RETURN_VALUE = 5
EXIT = 6
JUMP_IF_TRUE = 7
JUMP_IF_FALSE = 8
JUMP = 9


def get_location(pc, name, program):
    return "%s:%s:%s" % (name, pc, program.programs[name].get_instruction(pc))


jitdriver = JitDriver(greens=['pc', "name", 'program', ], reds=['machine'],
        get_printable_location=get_location)

class Bytecode(object):

    def __init__(self, instructions, literals, symbols):
        self.instructions = instructions # list of [op(int), ...]
        self.literals = literals # list of strings
        self.symbols = symbols # list of strings

    def get_instruction(self, pc):
        return self.instructions[pc]

    def get_symbol(self, index):
        return self.symbols[index]

    def get_literal(self, index):
        return self.literals[index]

class Program(object):

    def __init__(self):
        self.programs = {
            "main": Bytecode(
                instructions=[
                    [LOAD_LITERAL, 0], [PRINT], # print("hello world")
                    [LOAD_LITERAL, 0], [SAVE_VARIABLE, 0], # x = "hello world"
                    [LOAD_VARIABLE, 0], [PRINT], # print(x)
                    [LOAD_LITERAL, 1], [CALL_FUNCTION, 1, 1], [PRINT], # print(hello(x))
                    [LOAD_LITERAL, 2], [EXIT],
                ],
                literals=[
                    "hello world",
                    "world",
                    "2",
                ],
                symbols=[
                    "x",
                    "hello",
                ],
            ),
            "hello": Bytecode(
                instructions=[
                    [LOAD_VARIABLE, 0], [RETURN_VALUE], # return value
                ],
                literals=[],
                symbols=["value"]
            )
        }



def parse(source):
    return Program()

class Frame(object):
    def __init__(self, parent=None):
        self.parent = parent
        self.bindings = {}
    def save(self, key, value):
        self.bindings[key] = value
    def load(self, key):
        return self.bindings[key]


class Machine(object):

    def __init__(self, program):
        self.stack = []
        self.frame = Frame()
        self.running = True
        self.exit_code = 0

    def run_code(self, program, prog_name, pc):
        bytecode = program.programs[prog_name]
        inst = bytecode.get_instruction(pc)
        opcode = inst[0]
        if opcode == PRINT:
            print(self.stack.pop())
        elif opcode == LOAD_LITERAL:
            self.stack.append(bytecode.get_literal(inst[1]))
        elif opcode == LOAD_VARIABLE:
            self.stack.append(
                    self.frame.load(
                        bytecode.get_symbol(inst[1])))
        elif opcode == SAVE_VARIABLE:
            self.frame.save(bytecode.get_symbol(inst[1]),
                            self.stack.pop())
        elif opcode == JUMP:
            pc = inst[1]
        elif opcode == JUMP_IF_TRUE:
            val = self.stack.pop()
            if len(val) > 0:
                pc = inst[1]
        elif opcode == JUMP_IF_FALSE:
            val = self.stack.pop()
            if len(val) == 0:
                pc = inst[1]
        elif opcode == CALL_FUNCTION:
            prog_sym = bytecode.get_symbol(inst[1])
            func_bytecode = program.programs[prog_sym]
            frame = Frame(parent=self.frame)
            argc = inst[2]
            i = 0
            while i < argc:
                argv_i = func_bytecode.get_symbol(i)
                frame.save(argv_i, self.stack.pop())
                i = i + 1
            self.frame = frame
            self.stack.append(prog_name)
            self.stack.append(str(pc))
            return prog_sym, 0
        elif opcode == RETURN_VALUE:
            val = self.stack.pop()
            pc = int(self.stack.pop())
            prog_name = self.stack.pop()
            self.frame = self.frame.parent
            self.stack.append(val)
        elif opcode == EXIT:
            self.exit_code = int(self.stack.pop())
            self.running = False
        else:
            raise Exception("Unknown Bytecode")
        return prog_name, pc + 1


def mainloop(program):
    pc = 0
    name = "main"
    machine = Machine(program)
    while pc < len(program.programs[name].instructions) and machine.running:
        jitdriver.jit_merge_point(pc=pc,
                                  name=name,
                                  program=program,
                                  machine=machine)
        name, pc = machine.run_code(program, name, pc)
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
