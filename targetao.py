import sys

import os
import sys
import json

from rpython.rlib.parsing.ebnfparse import parse_ebnf, make_parse_function
from rpython.rlib.parsing.parsing import ParseError
from rpython.rlib.parsing.deterministic import LexerError

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
LOAD_FUNCTION = 2
LOAD_VARIABLE = 3
SAVE_VARIABLE = 4
CALL_FUNCTION = 5
RETURN_VALUE = 6
JUMP_IF_TRUE = 7
JUMP_IF_FALSE = 8
JUMP = 9
EXIT = 10
MAKE_ARRAY = 11
MAKE_OBJECT = 12


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

EBNF = """
STRING: "\\"[^\\\\"]*\\"";
NUMBER: "\-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][\+\-]?[0-9]+)?";
IGNORE: " |\n";
BOOLEAN: "true|false";
NULL: "null";
IDENTIFIER: "[a-zA-Z_][a-zA-Z0-9_]*";
main: >stmt*< [EOF];
value: <print> | <return> | <f> | <let> | <if> | <while> | <apply> | <IDENTIFIER> | <STRING> | <NUMBER> | <object> | <array> | <BOOLEAN> | <NULL>;
object: ["{"] (entry [","])* entry* ["}"];
array: ["["] (value [","])* value* ["]"];
entry: STRING [":"] value;
let: IDENTIFIER ["="] value;
apply: IDENTIFIER ["("] (value [","])* value* [")"];
f: ["f"] ["("] (IDENTIFIER [","])* IDENTIFIER* [")"] block;
if: ["if"] ["("] value [")"] block elif* else?;
elif: ["elif"] ["("] value [")"] block;
else: ["else"] block;
while: ["while"] ["("] value [")"] block;
block: ["{"] stmt* ["}"];
print: ["print"] value;
return: ["return"] value;
stmt: value [";"];
"""

regexes, rules, _to_ast = parse_ebnf(EBNF)
parse_ebnf = make_parse_function(regexes, rules, eof=True)
to_ast = _to_ast()

class Program(object):

    def __init__(self):
        self.f_counter = 0
        self.programs = {}

    def parse_main(self, source):
        self.programs["main"] = Bytecode([], [], [])
        try:
            tree = parse_ebnf(source)
            ast = to_ast.transform(tree)
        except ParseError as e:
            print(e.nice_error_message('main', source))
            return
        except LexerError as e:
            print(e.nice_error_message('main'))
            return
        self.scan_ast("main", ast)

    def scan_ast(self, target, ast):
        if ast.symbol == 'main':
            for stmt in ast.children:
                self.scan_ast(target, stmt)
        elif ast.symbol == 'stmt':
            self.scan_ast(target, ast.children[0])
        elif ast.symbol == 'print':
            self.scan_ast(target, ast.children[0])
            self.programs[target].instructions.append([PRINT])
        elif ast.symbol in ('NULL', 'STRING', 'BOOLEAN', 'NUMBER'):
            self.programs[target].literals.append(ast.additional_info)
            self.programs[target].instructions.append([
                LOAD_LITERAL, len(self.programs[target].literals) - 1])
        elif ast.symbol == 'IDENTIFIER':
            if ast.additional_info not in self.programs[target].symbols:
                self.programs[target].symbols.append(ast.additional_info)
            index = self.programs[target].symbols.index(ast.additional_info)
            self.programs[target].instructions.append([LOAD_VARIABLE, index])
        elif ast.symbol == 'let':
            identifier = ast.children[0]
            right = ast.children[1]
            self.scan_ast(target, right)
            if identifier.additional_info not in self.programs[target].symbols:
                self.programs[target].symbols.append(identifier.additional_info)
            index = self.programs[target].symbols.index(identifier.additional_info)
            self.programs[target].instructions.append([SAVE_VARIABLE, index])
        elif ast.symbol == 'return':
            self.scan_ast(target, ast.children[0])
            self.programs[target].instructions.append([RETURN_VALUE])
        elif ast.symbol == 'f':
            self.f_counter = self.f_counter + 1
            f_target = '@%d' % self.f_counter
            if f_target not in self.programs[target].symbols:
                self.programs[target].symbols.append(f_target)
            params = [p.additional_info for p in ast.children if p.symbol == 'IDENTIFIER']
            program = Bytecode(instructions=[], literals=[], symbols=params)
            self.programs[f_target] = program
            self.scan_ast(f_target, ast.children[len(params)])
            index = self.programs[target].symbols.index(f_target)
            self.programs[target].instructions.append([LOAD_FUNCTION, index])
        elif ast.symbol == 'block':
            for stmt in ast.children:
                self.scan_ast(target, stmt)
        elif ast.symbol == 'apply':
            for param in ast.children[1:]:
                self.scan_ast(target, param)
            f_var = ast.children[0]
            self.scan_ast(target, f_var) # load function first
            argc = len(ast.children[1:]) # call function with argc(nubmer of args)
            self.programs[target].instructions.append([CALL_FUNCTION, argc])
        elif ast.symbol == 'while':
            start = len(self.programs[target].instructions)
            predicate = self.scan_ast(target, ast.children[0])
            self.programs[target].instructions.append([JUMP_IF_TRUE, 0])
            index = len(self.programs[target].instructions) - 1
            block = self.scan_ast(target, ast.children[1])
            self.programs[target].instructions.append([JUMP, start])
            self.programs[target].instructions[index][1] = len(self.programs[target].instructions)
        elif ast.symbol == 'if':
            end_jumping_indexes = []
            predicate = self.scan_ast(target, ast.children[0])
            self.programs[target].instructions.append([JUMP_IF_FALSE, 0])
            next_branching_index = len(self.programs[target].instructions) - 1
            true_block = self.scan_ast(target, ast.children[1])
            self.programs[target].instructions.append([JUMP, 0])
            end_jumping_indexes.append(len(self.programs[target].instructions) - 1)
            self.programs[target].instructions[next_branching_index][1] = len(self.programs[target].instructions)
            for condition in ast.children[2:]:
                if condition.symbol == 'elif':
                    cond_predicate = self.scan_ast(target, condition.children[0])
                    self.programs[target].instructions.append([JUMP_IF_FALSE, 0])
                    next_branching_index = len(self.programs[target].instructions) - 1
                    cond_block = self.scan_ast(target, condition.children[1])
                    self.programs[target].instructions.append([JUMP, 0])
                    end_jumping_indexes.append(len(self.programs[target].instructions) - 1)
                    self.programs[target].instructions[next_branching_index][1] = len(self.programs[target].instructions)
                elif condition.symbol == 'else':
                    else_block = self.scan_ast(target, condition.children[0])
            for end_jumping_index in end_jumping_indexes:
                self.programs[target].instructions[end_jumping_index][1] = len(self.programs[target].instructions)
        elif ast.symbol == 'array':
            for child in ast.children:
                self.scan_ast(target, child)
            self.programs[target].instructions.append([MAKE_ARRAY, len(ast.children)])
        elif ast.symbol == 'object':
            for entry in ast.children:
                key = entry.children[0]
                self.scan_ast(target, key)
                value = entry.children[1]
                self.scan_ast(target, value)
            self.programs[target].instructions.append([MAKE_OBJECT, len(ast.children)])


def parse(source):
    program = Program()
    program.parse_main(source)
    return program

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
        #print prog_name, pc, inst, self.stack
        #import pdb;pdb.set_trace()
        opcode = inst[0]
        if opcode == PRINT:
            val = self.stack.pop()
            print(val)
        elif opcode == LOAD_LITERAL:
            self.stack.append(bytecode.get_literal(inst[1]))
        elif opcode == LOAD_VARIABLE:
            sym = bytecode.get_symbol(inst[1])
            if sym.startswith('@'): # function
                self.stack.append(sym)
            else:
                self.stack.append(self.frame.load(sym))
        elif opcode == LOAD_FUNCTION:
            sym = bytecode.get_symbol(inst[1])
            self.stack.append(sym)
        elif opcode == SAVE_VARIABLE:
            sym = bytecode.get_symbol(inst[1])
            if sym.startswith('@'): # function
                self.stack.append(sym)
            else:
                self.frame.save(bytecode.get_symbol(inst[1]),
                                self.stack.pop())
        elif opcode == MAKE_ARRAY:
            i, argc = 0, inst[1]
            args = [self.stack.pop() for _ in range(argc)]
            x = '['
            while i < argc:
                x += args[argc - i - 1]
                if i < argc - 1:
                    x += ','
                i = i + 1
            x += ']'
            self.stack.append(x)
        elif opcode == MAKE_OBJECT:
            i, argc = 0, inst[1]
            # [..., [value, key]]
            args = [[self.stack.pop(), self.stack.pop()] for _ in range(argc)]
            x = '{'
            while i < argc:
                x += args[argc - i - 1][1]
                x += ':'
                x += args[argc - i - 1][0]
                if i < argc - 1:
                    x += ','
                i = i + 1
            x += '}'
            self.stack.append(x)
        elif opcode == JUMP:
            pc = inst[1]
            return prog_name, pc
        elif opcode == JUMP_IF_TRUE:
            val = self.stack.pop()
            if val == 'true':
                pc = inst[1]
            else:
                pc = pc + 1
            return prog_name, pc
        elif opcode == JUMP_IF_FALSE:
            val = self.stack.pop()
            if val == 'false':
                pc = inst[1]
            else:
                pc = pc + 1
            return prog_name, pc
        elif opcode == CALL_FUNCTION:
            prog_sym = self.stack.pop()
            func_bytecode = program.programs[prog_sym]
            frame = Frame(parent=self.frame)
            argc = inst[1]
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
