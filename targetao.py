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
JUMP_IF_TRUE_AND_POP = 7
JUMP_IF_FALSE_AND_POP = 8
JUMP = 9
EXIT = 10
MAKE_ARRAY = 11
MAKE_OBJECT = 12
BINARY_ADD = 13
BINARY_SUB = 14
BINARY_MUL = 15
BINARY_DIV = 16
BINARY_MOD = 17
BINARY_LSHIFT = 18
BINARY_RSHIFT = 19
BINARY_AND = 20
BINARY_OR = 21
BINARY_XOR = 22
JUMP_IF_TRUE_OR_POP = 23
JUMP_IF_FALSE_OR_POP = 24
LOGICAL_NOT = 25
BINARY_EQ = 26
BINARY_NE = 27
BINARY_GT = 28
BINARY_GE = 29
BINARY_LT = 30
BINARY_LE = 31
BINARY_IN = 32
UNARY_NEGATIVE = 33
UNARY_POSITIVE = 34
UNARY_REVERSE = 35

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
object: ["{"] (entry [","])* entry* ["}"];
array: ["["] (expr [","])* expr* ["]"];
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
        elif ast.symbol == 'return':
            self.scan_ast(target, ast.children[0])
            self.programs[target].instructions.append([RETURN_VALUE])
        elif ast.symbol == 'let':
            identifier = ast.children[0]
            right = ast.children[1]
            self.scan_ast(target, right)
            if identifier.additional_info not in self.programs[target].symbols:
                self.programs[target].symbols.append(identifier.additional_info)
            index = self.programs[target].symbols.index(identifier.additional_info)
            self.programs[target].instructions.append([SAVE_VARIABLE, index])
        elif ast.symbol == 'if':
            end_jumping_indexes = []
            predicate = self.scan_ast(target, ast.children[0])
            self.programs[target].instructions.append([JUMP_IF_FALSE_AND_POP, 0])
            next_branching_index = len(self.programs[target].instructions) - 1
            true_block = self.scan_ast(target, ast.children[1])
            self.programs[target].instructions.append([JUMP, 0])
            end_jumping_indexes.append(len(self.programs[target].instructions) - 1)
            self.programs[target].instructions[next_branching_index][1] = len(self.programs[target].instructions)
            for condition in ast.children[2:]:
                if condition.symbol == 'elif':
                    cond_predicate = self.scan_ast(target, condition.children[0])
                    self.programs[target].instructions.append([JUMP_IF_FALSE_AND_POP, 0])
                    next_branching_index = len(self.programs[target].instructions) - 1
                    cond_block = self.scan_ast(target, condition.children[1])
                    self.programs[target].instructions.append([JUMP, 0])
                    end_jumping_indexes.append(len(self.programs[target].instructions) - 1)
                    self.programs[target].instructions[next_branching_index][1] = len(self.programs[target].instructions)
                elif condition.symbol == 'else':
                    else_block = self.scan_ast(target, condition.children[0])
            for end_jumping_index in end_jumping_indexes:
                self.programs[target].instructions[end_jumping_index][1] = len(self.programs[target].instructions)
        elif ast.symbol == 'while':
            start = len(self.programs[target].instructions)
            predicate = self.scan_ast(target, ast.children[0])
            self.programs[target].instructions.append([JUMP_IF_FALSE_AND_POP, 0])
            index = len(self.programs[target].instructions) - 1
            block = self.scan_ast(target, ast.children[1])
            self.programs[target].instructions.append([JUMP, start])
            self.programs[target].instructions[index][1] = len(self.programs[target].instructions)
        elif ast.symbol == 'apply':
            for param in ast.children[1:]:
                self.scan_ast(target, param)
            f_var = ast.children[0]
            self.scan_ast(target, f_var) # load function first
            argc = len(ast.children[1:]) # call function with argc(nubmer of args)
            self.programs[target].instructions.append([CALL_FUNCTION, argc])
        elif ast.symbol == 'expr':
            self.scan_ast(target, ast.children[0])
        elif ast.symbol == 'test':
            self.scan_ast(target, ast.children[0])
        elif ast.symbol == 'or_test':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                indexes = []
                for and_expr in ast.children[1:]:
                    self.scan_ast(target, and_expr)
                    self.programs[target].instructions.append([JUMP_IF_TRUE_OR_POP, 0])
                    indexes.append(len(self.programs[target].instructions) - 1)
                end = len(self.programs[target].instructions)
                for index in indexes:
                    self.programs[target].instructions[index][1] = end
        elif ast.symbol == 'and_test':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                indexes = []
                for and_expr in ast.children[1:]:
                    self.scan_ast(target, and_expr)
                    self.programs[target].instructions.append([JUMP_IF_FALSE_OR_POP, 0])
                    indexes.append(len(self.programs[target].instructions) - 1)
                end = len(self.programs[target].instructions)
                for index in indexes:
                    self.programs[target].instructions[index][1] = end
        elif ast.symbol == 'not_test':
            self.scan_ast(target, ast.children[0])
            if ast.children[0].additional_info == 'not_test':
                self.programs[target].instructions.append([LOGICAL_NOT])
        elif ast.symbol == 'comparison':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                self.scan_ast(target, ast.children[2])
                if ast.children[1].additional_info == '==':
                    self.programs[target].instructions.append([BINARY_EQ])
                elif ast.children[1].additional_info == '!=':
                    self.programs[target].instructions.append([BINARY_NE])
                elif ast.children[1].additional_info == '>':
                    self.programs[target].instructions.append([BINARY_GT])
                elif ast.children[1].additional_info== '>=':
                    self.programs[target].instructions.append([BINARY_GE])
                elif ast.children[1].additional_info == '<':
                    self.programs[target].instructions.append([BINARY_LT])
                elif ast.children[1].additional_info == '<=':
                    self.programs[target].instructions.append([BINARY_LE])
                elif ast.children[1].additional_info == 'in':
                    self.programs[target].instructions.append([BINARY_IN])
        elif ast.symbol == 'or_expr':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                for xor_expr in ast.children[1:]:
                    self.scan_ast(target, xor_expr)
                    self.programs[target].instructions.append([BINARY_OR])
        elif ast.symbol == 'xor_expr':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                for and_expr in ast.children[1:]:
                    self.scan_ast(target, and_expr)
                    self.programs[target].instructions.append([BINARY_XOR])
        elif ast.symbol == 'and_expr':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                for shift_expr in ast.children[1:]:
                    self.scan_ast(target, shift_expr)
                    self.programs[target].instructions.append([BINARY_AND])
        elif ast.symbol == 'shift_expr':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                for index in range(1, len(ast.children) - 1, 2):
                    self.scan_ast(target, ast.children[index + 1])
                    if ast.children[index].additional_info == '<<':
                        self.programs[target].instructions.append([BINARY_LSHIFT])
                    elif ast.children[index].additional_info == '>>':
                        self.programs[target].instructions.append([BINARY_RSHIFT])
                    else:
                        raise Exception('unknown operator: %s' % ast.children[index].additional_info)
        elif ast.symbol == 'arith_expr':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                for index in range(1, len(ast.children) - 1, 2):
                    self.scan_ast(target, ast.children[index + 1])
                    if ast.children[index].additional_info == '+':
                        self.programs[target].instructions.append([BINARY_ADD])
                    elif ast.children[index].additional_info == '-':
                        self.programs[target].instructions.append([BINARY_SUB])
                    else:
                        raise Exception('unknown operator: %s' % ast.children[index].additional_info)
        elif ast.symbol == 'term':
            self.scan_ast(target, ast.children[0])
            if len(ast.children) > 1:
                for index in range(1, len(ast.children) - 1, 2):
                    self.scan_ast(target, ast.children[index + 1])
                    if ast.children[index].additional_info == '*':
                        self.programs[target].instructions.append([BINARY_MUL])
                    elif ast.children[index].additional_info == '/':
                        self.programs[target].instructions.append([BINARY_DIV])
                    elif ast.children[index].additional_info == '%':
                        self.programs[target].instructions.append([BINARY_MOD])
                    else:
                        raise Exception('unknown operator: %s' % ast.children[index].additional_info)
        elif ast.symbol == 'factor':
            if len(ast.children) == 2:
                self.scan_ast(target, ast.children[1])
                if ast.children[0].additional_info == '+':
                    self.programs[target].instructions.append([UNARY_POSITIVE])
                elif ast.children[0].additional_info == '-':
                    self.programs[target].instructions.append([UNARY_NEGATIVE])
                elif ast.children[0].additional_info == '~':
                    self.programs[target].instructions.append([UNARY_REVERSE])
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
        elif ast.symbol == 'IDENTIFIER':
            if ast.additional_info not in self.programs[target].symbols:
                self.programs[target].symbols.append(ast.additional_info)
            index = self.programs[target].symbols.index(ast.additional_info)
            self.programs[target].instructions.append([LOAD_VARIABLE, index])
        elif ast.symbol in ('NULL', 'STRING', 'BOOLEAN', 'NUMBER'):
            self.programs[target].literals.append(ast.additional_info)
            self.programs[target].instructions.append([
                LOAD_LITERAL, len(self.programs[target].literals) - 1])
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
        elif ast.symbol == 'block':
            for stmt in ast.children:
                self.scan_ast(target, stmt)

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
        #print prog_name, pc, inst, self.stack, bytecode.instructions
        #import pdb;pdb.set_trace()
        opcode = inst[0]
        if opcode == PRINT:
            print(self.stack.pop())
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
        elif opcode == BINARY_ADD:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append(str(int(left) + int(right)))
        elif opcode == BINARY_SUB:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append(str(int(left) - int(right)))
        elif opcode == BINARY_MUL:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append(str(int(left) * int(right)))
        elif opcode == BINARY_DIV:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append(str(int(left) / int(right)))
        elif opcode == BINARY_MOD:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append(str(int(left) % int(right)))
        elif opcode == BINARY_LSHIFT:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append(str(int(left) << int(right)))
        elif opcode == BINARY_RSHIFT:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append(str(int(left) >> int(right)))
        elif opcode == BINARY_AND:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append(str(int(left) & int(right)))
        elif opcode == BINARY_OR:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append(str(int(left) | int(right)))
        elif opcode == BINARY_XOR:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append(str(int(left) ^ int(right)))
        elif opcode == BINARY_EQ: # FIXME: support for all types.
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append('true' if int(left) == int(right) else 'false')
        elif opcode == BINARY_NE:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append('true' if int(left) != int(right) else 'false')
        elif opcode == BINARY_GT:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append('true' if int(left) > int(right) else 'false')
        elif opcode == BINARY_GE:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append('true' if int(left) >= int(right) else 'false')
        elif opcode == BINARY_LT:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append('true' if int(left) < int(right) else 'false')
        elif opcode == BINARY_LE:
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append('true' if int(left) <= int(right) else 'false')
        elif opcode == BINARY_IN: # FIXME: won't work.
            right, left = self.stack.pop(), self.stack.pop()
            self.stack.append('true' if left in right else 'false')
        elif opcode == BINARY_NE:
            pass
        elif opcode == UNARY_NEGATIVE:
            value = self.stack.pop()
            self.stack.append(str(-1 * int(value)))
        elif opcode == UNARY_REVERSE:
            value = self.stack.pop()
            self.stack.append(str(~int(value)))
        elif opcode == JUMP:
            pc = inst[1]
            return prog_name, pc
        elif opcode == JUMP_IF_TRUE_AND_POP:
            val = self.stack.pop()
            if val == 'true':
                pc = inst[1]
            else:
                pc = pc + 1
            return prog_name, pc
        elif opcode == JUMP_IF_FALSE_AND_POP:
            val = self.stack.pop()
            if val == 'false':
                pc = inst[1]
            else:
                pc = pc + 1
            return prog_name, pc
        elif opcode == JUMP_IF_TRUE_OR_POP:
            val = self.stack[-1]
            if val == 'true':
                pc = inst[1]
            else:
                self.stack.pop()
                pc = pc + 1
            return prog_name, pc
        elif opcode == JUMP_IF_FALSE_OR_POP:
            val = self.stack[-1]
            if val == 'false':
                pc = inst[1]
            else:
                self.stack.pop()
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
