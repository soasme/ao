import os
import sys
import json

from rpython.rlib.parsing.ebnfparse import parse_ebnf, make_parse_function
from rpython.rlib.parsing.parsing import ParseError
from rpython.rlib.parsing.deterministic import LexerError
from rpython.rlib.jit import JitDriver, purefunction

from libs.builtin import MODULE

EXIT = 0;               PRINT = 1;              LOAD_LITERAL = 2;
LOAD_FUNCTION = 3;      LOAD_VARIABLE = 4;      SAVE_VARIABLE = 5;
CALL_FUNCTION = 6;      RETURN_VALUE = 7;       MAKE_ARRAY = 8;
MAKE_OBJECT = 9;        JUMP = 10;              JUMP_IF_TRUE_AND_POP = 11;
BINARY_ADD = 12;        BINARY_SUB = 13;        JUMP_IF_FALSE_AND_POP = 14;
BINARY_MUL = 15;        BINARY_DIV = 16;        JUMP_IF_TRUE_OR_POP = 17;
BINARY_MOD = 18;        BINARY_LSHIFT = 19;     JUMP_IF_FALSE_OR_POP = 20;
BINARY_LSHIFT = 21;     BINARY_RSHIFT = 22;     BINARY_AND = 23;
BINARY_OR = 24;         BINARY_XOR = 25;        BINARY_EQ = 26;
BINARY_NE = 27;         BINARY_GT = 28;         BINARY_GE = 29;
BINARY_LT = 30;         BINARY_LE = 31;         BINARY_IN = 32;
UNARY_NEGATIVE = 33;    UNARY_POSITIVE = 34;    UNARY_REVERSE = 35;
LOGICAL_NOT = 36;

CODES = """
    EXIT            PRINT             LOAD_LITERAL
    LOAD_FUNCTION   LOAD_VARIABLE     SAVE_VARIABLE
    CALL_FUNCTION   RETURN_VALUE      MAKE_ARRAY
    MAKE_OBJECT     JUMP              JUMP_IF_TRUE_AND_POP
    BINARY_ADD      BINARY_SUB        JUMP_IF_FALSE_AND_POP
    BINARY_MUL      BINARY_DIV        JUMP_IF_TRUE_OR_POP
    BINARY_MOD      BINARY_LSHIFT     JUMP_IF_FALSE_OR_POP
    BINARY_LSHIFT   BINARY_RSHIFT     BINARY_AND
    BINARY_OR       BINARY_XOR        BINARY_EQ
    BINARY_NE       BINARY_GT         BINARY_GE
    BINARY_LT       BINARY_LE         BINARY_IN
    UNARY_NEGATIVE  UNARY_POSITIVE    UNARY_REVERSE
    LOGICAL_NOT
""".split()

BINARY_OP = {
    '==': BINARY_EQ, '!=': BINARY_NE, '>': BINARY_GT, '>=': BINARY_GE,
    '<': BINARY_LT, '<=': BINARY_LE, 'in': BINARY_IN, '<<': BINARY_LSHIFT,
    '>>': BINARY_RSHIFT, '+': BINARY_ADD, '-': BINARY_SUB, '*': BINARY_MUL,
    '/': BINARY_DIV, '%': BINARY_MOD,
}

def get_location(pc, name, program):
    lineno = program.programs[name].get_pc_position(pc)
    source = program.programs[name].source.splitlines()[lineno]
    instruction = program.programs[name].instructions[pc]
    op = CODES[instruction[0]]
    return "Code %s, line %d\n  %s\n  INSTRUCTION: %s %s" % (name, lineno+1, source, op, instruction)

jitdriver = JitDriver(greens=['pc', "name", 'program', ], reds=['machine'],
        get_printable_location=get_location)


class Bytecode(object):

    def __init__(self, instructions, literals, symbols, filename=None, source=None):
        self.instructions = instructions # list of [op(int), ...]
        self.literals = literals # list of strings
        self.symbols = symbols # list of strings
        self.source = source
        self.filename = filename
        self.positions = {}

    def get_instruction(self, pc):
        return self.instructions[pc]

    def get_symbol(self, index):
        return self.symbols[index]

    def get_literal(self, index):
        return self.literals[index]

    def emit(self, instruction, ast):
        self.instructions.append(instruction)
        self.positions[len(self.instructions) - 1] = ast.getsourcepos().lineno

    def get_pc_position(self, pc):
        return self.positions[pc]

EBNF = """
STRING: "\\"[^\\\\"]*\\"";
NUMBER: "\-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][\+\-]?[0-9]+)?";
IGNORE: " |\n";
BOOLEAN: "true|false";
NULL: "null";
IDENTIFIER: "[a-zA-Z_][a-zA-Z0-9_]*";
main: >stmt*< [EOF];
expr: <test>;
test: <or_test>;
or_test: and_test >(["or"] and_test)+< | <and_test>;
and_test: not_test >(["and"] not_test)+< | <not_test>;
not_test: ["not"] not_test | <comparison>;
comparison: or_expr comparison_op or_expr | <or_expr>;
comparison_op: <"=="> | <"!="> | <"<"> | <"<="> | <">"> | <">="> | <"in">;
or_expr: xor_expr >(["|"] xor_expr)+< | <xor_expr>;
xor_expr: and_expr >(["^"] and_expr)+< | <and_expr>;
and_expr: shift_expr >(["&"] shift_expr)+< | <shift_expr>;
shift_expr: arith_expr >(shift_op arith_expr)+< | <arith_expr>;
shift_op: <"<<"> | <">>">;
arith_expr: term >(arith_op term)+< | <term>;
arith_op: <"+"> | <"-">;
term: factor >(term_op factor)+< | <factor>;
term_op: <"*"> | <"/"> | <"%">;
factor: factor_op factor | <primary_expression>;
factor_op: <"+"> | <"-"> | <"~">;
primary_expression: <f> | <apply> | <IDENTIFIER> | <STRING> | <NUMBER> | <object> | <array> | <BOOLEAN> | <NULL>;
object: "{" (entry [","])* entry* ["}"];
array: "[" (expr [","])* expr* ["]"];
entry: STRING [":"] expr;
let: IDENTIFIER ["="] expr;
apply: IDENTIFIER ["("] (expr [","])* expr* [")"];
f: ["f"] ["("] (IDENTIFIER [","])* IDENTIFIER* [")"] block;
if: ["if"] ["("] expr [")"] block elif* else?;
elif: ["elif"] ["("] expr [")"] block;
else: ["else"] block;
while: ["while"] ["("] expr [")"] block;
block: ["{"] stmt* ["}"];
print: ["print"] expr;
return: ["return"] expr;
stmt: <print> [";"] | <return> [";"] | <let> [";"] | <if> | <while> | <apply> [";"];
"""

regexes, rules, _to_ast = parse_ebnf(EBNF)
parse_ebnf = make_parse_function(regexes, rules, eof=True)
to_ast = _to_ast()

class Program(object):

    def __init__(self):
        self.f_counter = 0
        self.programs = {}
        self.preludes = {}

    def remove_comment(self, source):
        lines = source.splitlines()
        return '\n'.join([l for l in lines if not l.lstrip().startswith('#')])

    def parse_main(self, filename, source):
        try:
            source = self.remove_comment(source)
            tree = parse_ebnf(source)
            ast = to_ast.transform(tree)
        except ParseError as e:
            print(e.nice_error_message(filename, source))
        except LexerError as e:
            print(e.nice_error_message(filename))
        else:
            target = filename
            self.programs[target] = Bytecode([], [], [], filename=target, source=source)
            self.scan_ast(target, ast)

    def scan_ast(self, target, ast):
        if ast.symbol == 'main':
            for stmt in ast.children:
                self.scan_ast(target, stmt)
        elif ast.symbol == 'stmt':
            self.scan_ast(target, ast.children[0])
        elif ast.symbol == 'print':
            self.scan_ast(target, ast.children[0])
            self.programs[target].emit([PRINT], ast)
        elif ast.symbol == 'return':
            self.scan_ast(target, ast.children[0])
            self.programs[target].emit([RETURN_VALUE], ast)
        elif ast.symbol == 'let':
            identifier = ast.children[0]
            right = ast.children[1]
            self.scan_ast(target, right)
            if identifier.additional_info not in self.programs[target].symbols:
                self.programs[target].symbols.append(identifier.additional_info)
            index = self.programs[target].symbols.index(identifier.additional_info)
            self.programs[target].emit([SAVE_VARIABLE, index], ast)
        elif ast.symbol == 'if':
            end_jumping_indexes = []
            predicate = self.scan_ast(target, ast.children[0])
            self.programs[target].emit([JUMP_IF_FALSE_AND_POP, 0], ast)
            next_branching_index = len(self.programs[target].instructions) - 1
            true_block = self.scan_ast(target, ast.children[1])
            self.programs[target].emit([JUMP, 0], ast)
            end_jumping_indexes.append(len(self.programs[target].instructions) - 1)
            self.programs[target].instructions[next_branching_index][1] = len(self.programs[target].instructions)
            for condition in ast.children[2:]:
                if condition.symbol == 'elif':
                    cond_predicate = self.scan_ast(target, condition.children[0])
                    self.programs[target].emit([JUMP_IF_FALSE_AND_POP, 0], ast)
                    next_branching_index = len(self.programs[target].instructions) - 1
                    cond_block = self.scan_ast(target, condition.children[1])
                    self.programs[target].emit([JUMP, 0], ast)
                    end_jumping_indexes.append(len(self.programs[target].instructions) - 1)
                    self.programs[target].instructions[next_branching_index][1] = len(self.programs[target].instructions)
                elif condition.symbol == 'else':
                    else_block = self.scan_ast(target, condition.children[0])
            for end_jumping_index in end_jumping_indexes:
                self.programs[target].instructions[end_jumping_index][1] = len(self.programs[target].instructions)
        elif ast.symbol == 'while':
            start = len(self.programs[target].instructions)
            predicate = self.scan_ast(target, ast.children[0])
            self.programs[target].emit([JUMP_IF_FALSE_AND_POP, 0], ast)
            index = len(self.programs[target].instructions) - 1
            block = self.scan_ast(target, ast.children[1])
            self.programs[target].emit([JUMP, start], ast)
            self.programs[target].instructions[index][1] = len(self.programs[target].instructions)
        elif ast.symbol == 'apply':
            for param in ast.children[1:]:
                self.scan_ast(target, param)
            f_var = ast.children[0]
            self.scan_ast(target, f_var) # load function first
            argc = len(ast.children[1:]) # call function with argc(nubmer of args)
            self.programs[target].emit([CALL_FUNCTION, argc], ast)
        elif ast.symbol == 'expr':
            self.scan_ast(target, ast.children[0])
        elif ast.symbol == 'test':
            self.scan_ast(target, ast.children[0])
        elif ast.symbol == 'or_test':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                indexes = []
                for and_expr in ast.children[1:]:
                    self.programs[target].emit([JUMP_IF_TRUE_OR_POP, 0], ast)
                    indexes.append(len(self.programs[target].instructions) - 1)
                    self.scan_ast(target, and_expr)
                end = len(self.programs[target].instructions)
                for index in indexes:
                    self.programs[target].instructions[index][1] = end
        elif ast.symbol == 'and_test':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                indexes = []
                for and_expr in ast.children[1:]:
                    self.programs[target].emit([JUMP_IF_FALSE_OR_POP, 0], ast)
                    indexes.append(len(self.programs[target].instructions) - 1)
                    self.scan_ast(target, and_expr)
                end = len(self.programs[target].instructions)
                for index in indexes:
                    self.programs[target].instructions[index][1] = end
        elif ast.symbol == 'not_test':
            self.scan_ast(target, ast.children[0])
            self.programs[target].emit([LOGICAL_NOT], ast)
        elif ast.symbol == 'comparison':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                self.scan_ast(target, ast.children[2])
                op = BINARY_OP[ast.children[1].additional_info]
                self.programs[target].emit([op], ast)
        elif ast.symbol == 'or_expr':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                for xor_expr in ast.children[1:]:
                    self.scan_ast(target, xor_expr)
                    self.programs[target].emit([BINARY_OR], ast)
        elif ast.symbol == 'xor_expr':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                for and_expr in ast.children[1:]:
                    self.scan_ast(target, and_expr)
                    self.programs[target].emit([BINARY_XOR], ast)
        elif ast.symbol == 'and_expr':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                for shift_expr in ast.children[1:]:
                    self.scan_ast(target, shift_expr)
                    self.programs[target].emit([BINARY_AND], ast)
        elif ast.symbol == 'shift_expr':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                for index in range(1, len(ast.children) - 1, 2):
                    self.scan_ast(target, ast.children[index + 1])
                    op = BINARY_OP[ast.children[index].additional_info]
                    self.programs[target].emit([op], ast)
        elif ast.symbol == 'arith_expr':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                for index in range(1, len(ast.children) - 1, 2):
                    self.scan_ast(target, ast.children[index + 1])
                    op = BINARY_OP[ast.children[index].additional_info]
                    self.programs[target].emit([op], ast)
        elif ast.symbol == 'term':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                for index in range(1, len(ast.children) - 1, 2):
                    self.scan_ast(target, ast.children[index + 1])
                    op = BINARY_OP[ast.children[index].additional_info]
                    self.programs[target].emit([op], ast)
        elif ast.symbol == 'factor':
            if len(ast.children) == 2:
                self.scan_ast(target, ast.children[1])
                if ast.children[0].additional_info == '+':
                    self.programs[target].emit([UNARY_POSITIVE], ast)
                elif ast.children[0].additional_info == '-':
                    self.programs[target].emit([UNARY_NEGATIVE], ast)
                elif ast.children[0].additional_info == '~':
                    self.programs[target].emit([UNARY_REVERSE], ast)
        elif ast.symbol == 'f':
            self.f_counter = self.f_counter + 1
            f_target = '@%d' % self.f_counter
            if f_target not in self.programs[target].symbols:
                self.programs[target].symbols.append(f_target)
            params = [p.additional_info for p in ast.children if p.symbol == 'IDENTIFIER']
            program = Bytecode(instructions=[], literals=[], symbols=params,
                    filename=target, source=self.programs[target].source)
            self.programs[f_target] = program
            self.scan_ast(f_target, ast.children[len(params)])
            if program.instructions[-1][0] != RETURN_VALUE: program.emit([RETURN_VALUE], ast)
            index = self.programs[target].symbols.index(f_target)
            self.programs[target].emit([LOAD_FUNCTION, index], ast)
        elif ast.symbol == 'IDENTIFIER':
            if ast.additional_info not in self.programs[target].symbols:
                self.programs[target].symbols.append(ast.additional_info)
            index = self.programs[target].symbols.index(ast.additional_info)
            self.programs[target].emit([LOAD_VARIABLE, index], ast)
        elif ast.symbol in ('NULL', 'STRING', 'BOOLEAN', 'NUMBER'):
            self.programs[target].literals.append(ast.additional_info)
            self.programs[target].emit([
                LOAD_LITERAL, len(self.programs[target].literals) - 1], ast)
        elif ast.symbol == 'array':
            for child in ast.children[1:]:
                self.scan_ast(target, child)
            self.programs[target].emit([MAKE_ARRAY, len(ast.children[1:])], ast)
        elif ast.symbol == 'object':
            for entry in ast.children[1:]:
                key = entry.children[0]
                self.scan_ast(target, key)
                value = entry.children[1]
                self.scan_ast(target, value)
            self.programs[target].emit([MAKE_OBJECT, len(ast.children[1:])], ast)
        elif ast.symbol == 'block':
            for stmt in ast.children:
                self.scan_ast(target, stmt)

class Code(object):

    def __init__(self, name, environment_frame):
        self.name = name
        self.frame = environment_frame

class Frame(object):
    def __init__(self, name, pc=0, parent=None, bytecode=None):
        self.bindings = {}
        self.evaluations = []
        self.codes = {}
        self.parent = parent
        self.pc = pc
        self.name = name
        self.bytecode = bytecode
    def save_pc(self, pc):
        self.pc = pc
    def save(self, key, value, frame=None):
        self.bindings[key] = value
        if value.startswith('@') and value not in self.codes:
            self.codes[value] = Code(value, frame)
    def load(self, key):
        if key in self.bindings:
            return self.bindings[key]
        elif self.parent is not None:
            return self.parent.load(key)
        elif key in MODULE:
            return '@@' + key
        else:
            raise Exception('unknown variable: %s' % key)
    def get_code(self, key):
        if key in self.codes:
            return self.codes[key]
        elif key not in self.codes and self.parent is None:
            raise Exception('unknown function: %s' % key)
        else:
            return self.parent.get_code(key)
    def push(self, value):
        self.evaluations.append(value)
    def pop(self):
        return self.evaluations.pop()
    def peek(self):
        return self.evaluations[-1]
    def str(self):
        return '%s' % self.name

class BuiltinContext(object):
    def __init__(self, machine, tos, bytecode):
        self.machine = machine
        self.tos = tos
        self.bytecode = bytecode

class Machine(object):

    def __init__(self, entry, program):
        self.running = True
        self.exit_code = 0
        self.program = program
        self.error = None
        self.stack = [Frame(entry, bytecode=program.programs[entry])]

    def run_code(self, prog_name, pc):
        tos = self.stack[-1]
        tos.save_pc(pc)
        bytecode = self.program.programs[prog_name]
        inst = bytecode.get_instruction(pc)
        opcode = inst[0]
        #print prog_name, pc, inst, tos.evaluations
        print get_location(pc, prog_name, self.program)
        if opcode == CALL_FUNCTION:
            func_sym = tos.pop()
            if func_sym.startswith('@@'):
                builtin = MODULE[func_sym[2:]]
                builtin(BuiltinContext(self, tos, bytecode))
            else:
                func_code = tos.get_code(func_sym)
                parent_frame = func_code.frame
                func_bytecode = self.program.programs[func_sym]
                new_frame = Frame(name=func_sym, pc=0, parent=parent_frame, bytecode=func_bytecode)
                argc = inst[1]
                for i in range(argc):
                    argv_i_sym = func_bytecode.get_symbol(i)
                    new_frame.save(argv_i_sym, tos.pop(), new_frame)
                self.stack.append(new_frame)
                return func_sym, 0
        elif opcode == RETURN_VALUE:
            val = tos.pop()
            old_frame = self.stack.pop()
            tos = self.stack[-1]
            pc = tos.pc
            prog_name = tos.name
            tos.push(val)
            if val.startswith('@'):
                tos.codes[val] = Code(val, old_frame)
        elif opcode == PRINT:
            print(tos.pop())
        elif opcode == LOAD_LITERAL:
            tos.push(bytecode.get_literal(inst[1]))
        elif opcode == LOAD_VARIABLE:
            sym = bytecode.get_symbol(inst[1])
            tos.push(tos.load(sym))
        elif opcode == LOAD_FUNCTION:
            sym = bytecode.get_symbol(inst[1])
            tos.push(sym)
        elif opcode == SAVE_VARIABLE:
            sym = bytecode.get_symbol(inst[1])
            tos.save(sym, tos.pop(), tos)
        elif opcode == MAKE_ARRAY:
            argc, args, x = inst[1], [tos.pop() for _ in range(inst[1])], '['
            for i in range(argc):
                x += args[argc - i - 1]
                if i < argc - 1:
                    x += ','
            x += ']'
            tos.push(x)
        elif opcode == MAKE_OBJECT:
            i, argc = 0, inst[1]
            # [..., [value, key]]
            args = [[tos.pop(), tos.pop()] for _ in range(argc)]
            x = '{'
            while i < argc:
                x += args[argc - i - 1][1]
                x += ':'
                x += args[argc - i - 1][0]
                if i < argc - 1:
                    x += ','
                i = i + 1
            x += '}'
            tos.push(x)
        elif opcode == BINARY_ADD:
            MODULE['add'](BuiltinContext(self, tos, bytecode))
        elif opcode == BINARY_SUB:
            right, left = tos.pop(), tos.pop()
            tos.push(str(int(left) - int(right)))
        elif opcode == BINARY_MUL:
            right, left = tos.pop(), tos.pop()
            tos.push(str(int(left) * int(right)))
        elif opcode == BINARY_DIV:
            right, left = tos.pop(), tos.pop()
            tos.push(str(int(left) / int(right)))
        elif opcode == BINARY_MOD:
            right, left = tos.pop(), tos.pop()
            tos.push(str(int(left) % int(right)))
        elif opcode == BINARY_LSHIFT:
            right, left = tos.pop(), tos.pop()
            tos.push(str(int(left) << int(right)))
        elif opcode == BINARY_RSHIFT:
            right, left = tos.pop(), tos.pop()
            tos.push(str(int(left) >> int(right)))
        elif opcode == BINARY_AND:
            right, left = tos.pop(), tos.pop()
            tos.push(str(int(left) & int(right)))
        elif opcode == BINARY_OR:
            right, left = tos.pop(), tos.pop()
            tos.push(str(int(left) | int(right)))
        elif opcode == BINARY_XOR:
            right, left = tos.pop(), tos.pop()
            tos.push(str(int(left) ^ int(right)))
        elif opcode == BINARY_EQ: # FIXME: support for all types.
            MODULE['eq'](BuiltinContext(self, tos, bytecode))
        elif opcode == BINARY_NE:
            right, left = tos.pop(), tos.pop()
            tos.push('true' if left != right else 'false')
        elif opcode == BINARY_GT:
            right, left = tos.pop(), tos.pop()
            tos.push('true' if left > right else 'false')
        elif opcode == BINARY_GE:
            right, left = tos.pop(), tos.pop()
            tos.push('true' if left >= right else 'false')
        elif opcode == BINARY_LT:
            right, left = tos.pop(), tos.pop()
            tos.push('true' if left < right else 'false')
        elif opcode == BINARY_LE:
            right, left = tos.pop(), tos.pop()
            tos.push('true' if left <= right else 'false')
        elif opcode == BINARY_IN: # FIXME: won't work.
            right, left = tos.pop(), tos.pop()
            tos.push('true' if left in right else 'false')
        elif opcode == BINARY_NE:
            pass
        elif opcode == UNARY_NEGATIVE:
            value = tos.pop()
            tos.push(str(-1 * int(value)))
        elif opcode == UNARY_REVERSE:
            value = tos.pop()
            tos.push(str(~int(value)))
        elif opcode == LOGICAL_NOT:
            value = tos.pop()
            tos.push('true' if value == 'false' or value == '[]' or value == '{}'
                    or value == '""' or value == '0' or value == '0.0'else 'false')
        elif opcode == JUMP:
            pc = inst[1]
            return prog_name, pc
        elif opcode == JUMP_IF_TRUE_AND_POP:
            val = tos.pop()
            if val == 'true':
                pc = inst[1]
            else:
                pc = pc + 1
            return prog_name, pc
        elif opcode == JUMP_IF_FALSE_AND_POP:
            val = tos.pop()
            if val == 'false':
                pc = inst[1]
            else:
                pc = pc + 1
            return prog_name, pc
        elif opcode == JUMP_IF_TRUE_OR_POP:
            val = tos.peek()
            if val == 'true':
                pc = inst[1]
            else:
                tos.pop()
                pc = pc + 1
            return prog_name, pc
        elif opcode == JUMP_IF_FALSE_OR_POP:
            val = tos.peek()
            if val == 'false':
                pc = inst[1]
            else:
                tos.pop()
                pc = pc + 1
            return prog_name, pc
        elif opcode == EXIT:
            self.exit_code = int(tos.pop())
            self.running = False
        else:
            raise Exception("Unknown Bytecode")
        return prog_name, pc + 1

def crash_stack(machine):
    print >> sys.stderr, 'Traceback: %s' % machine.error
    for frame in machine.stack:
        lineno = frame.bytecode.get_pc_position(frame.pc)
        print >> sys.stderr, 'File %s, line %d' % (frame.str(), lineno)
        print >> sys.stderr, '  %s' % frame.bytecode.source.splitlines()[lineno]
    machine.exit_code = 1
    machine.running = False

def mainloop(filename, program):
    pc = 0
    name = filename
    machine = Machine(name, program)
    while pc < len(program.programs[name].instructions) and machine.running:
        jitdriver.jit_merge_point(pc=pc, name=name, program=program, machine=machine)
        name, pc = machine.run_code(name, pc)
        if machine.error is not None:
            crash_stack(machine)
            break
    return machine.exit_code

def run(filename):
    program_contents = ""
    try:
        fp = os.open(filename, os.O_RDONLY, 0777)
    except OSError as e:
        print >> sys.stderr, str(e)
        return 1
    while True:
        read = os.read(fp, 4096)
        if len(read) == 0: break
        program_contents += read
    os.close(fp)
    program = Program()
    program.parse_main(filename, program_contents)
    return mainloop(filename, program)

def entry_point(argv):
    try:
        filename = argv[1]
    except IndexError:
        print >> sys.stderr, "You must supply a filename"
        return 1
    return run(filename)

def target(driver, *args):
    driver.exe_name = 'ao'
    return entry_point, None

def jitpolicy(driver):
    from rpython.jit.codewriter.policy import JitPolicy
    return JitPolicy()

if __name__ == "__main__":
    entry_point(sys.argv)
