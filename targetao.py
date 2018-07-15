import os
import sys
import json
from rpython.rlib.parsing.ebnfparse import parse_ebnf, make_parse_function
from rpython.rlib.parsing.parsing import ParseError
from rpython.rlib.parsing.deterministic import LexerError
from rpython.rlib.parsing.tree import RPythonVisitor
from rpython.rlib.objectmodel import not_rpython
from rpython.rlib.jit import JitDriver, purefunction
from rpython.rlib.rbigint import rbigint
from builtin import MODULE

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
LOGICAL_NOT = 36;       CATCH_ERROR = 37;

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
    LOGICAL_NOT     CATCH_ERROR
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
        try:
            self.positions[len(self.instructions) - 1] = ast.getsourcepos().lineno
        except IndexError:
            self.positions[len(self.instructions) - 1] = -1

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
catch: IDENTIFIER ["or"] IDENTIFIER ["="] expr;
apply: IDENTIFIER ["("] (expr [","])* expr* [")"];
f: ["f"] ["("] (IDENTIFIER [","])* IDENTIFIER* [")"] block;
if: ["if"] ["("] expr [")"] block elif* else?;
elif: ["elif"] ["("] expr [")"] block;
else: ["else"] block;
while: ["while"] ["("] expr [")"] block;
block: ["{"] stmt* ["}"];
print: ["print"] expr;
return: ["return"] expr;
stmt: <print> [";"] | <return> [";"] | <catch> [";"] | <let> [";"] | <if> | <while> | <apply> [";"];
"""

regexes, rules, _to_ast = parse_ebnf(EBNF)
parse_ebnf = make_parse_function(regexes, rules, eof=True)
to_ast = _to_ast()

class Compiler(RPythonVisitor):

    def __init__(self, target, program=None):
        self.f_counter = 0
        self.target = target
        self.programs = program or {}

    def remove_comment(self, source):
        lines = source.splitlines()
        return '\n'.join([l for l in lines if not l.lstrip().startswith('#')])

    def compile(self, filename, source):
        try:
            source = self.remove_comment(source)
            tree = parse_ebnf(source)
            ast = to_ast.transform(tree)
        except ParseError as e:
            print(e.nice_error_message(filename, source))
        except LexerError as e:
            print(e.nice_error_message(filename))
        else:
            self.target = filename
            self.programs[self.target] = Bytecode([], [], [],
                    filename=filename, source=source)
            self.dispatch(ast)

    def emit(self, inst, ast):
        self.programs[self.target].emit(inst, ast)

    def _visit_keyword_expr(self, inst, ast):
        self.dispatch(ast.children[0]); self.emit(inst, ast)

    def _get_identifier_index(self, identifier):
        if identifier.additional_info not in self.programs[self.target].symbols:
            self.programs[self.target].symbols.append(identifier.additional_info)
        return self.programs[self.target].symbols.index(identifier.additional_info)

    def _emit_future(self, ast):
        self.emit([0, 0], ast)
        return len(self.programs[self.target].instructions) - 1

    def _set_future(self, index, inst):
        self.programs[self.target].instructions[index] = inst

    def _get_instructions_size(self):
        return len(self.programs[self.target].instructions)

    def visit_main(self, ast):
        for stmt in ast.children: self.dispatch(stmt)

    def visit_stmt(self, ast): self.dispatch(ast.children[0])

    def visit_print(self, ast): self._visit_keyword_expr([PRINT], ast)

    def visit_return(self, ast): self._visit_keyword_expr([RETURN_VALUE], ast)

    def visit_let(self, ast):
        self.dispatch(ast.children[1])
        self.emit([SAVE_VARIABLE, self._get_identifier_index(ast.children[0])], ast)

    def visit_if(self, ast):
        ends = []
        self.dispatch(ast.children[0])
        pred_fut = self._emit_future(ast)
        self.dispatch(ast.children[1])
        ends.append(self._emit_future(ast))
        self._set_future(pred_fut, [JUMP_IF_FALSE_AND_POP, self._get_instructions_size()])
        for cond in ast.children[2:]:
            if cond.symbol == 'elif':
                self.dispatch(cond.children[0])
                pred_fut = self._emit_future(ast)
                self.dispatch(cond.children[1])
                ends.append(self._emit_future(ast))
                self._set_future(pred_fut,
                        [JUMP_IF_FALSE_AND_POP, self._get_instructions_size()])
            else:
                self.dispatch(cond.children[0])
        for end in ends:
            self._set_future(end, [JUMP, self._get_instructions_size()])

    def visit_while(self, ast):
        start = self._get_instructions_size()
        self.dispatch(ast.children[0])
        pred_fut = self._emit_future(ast)
        self.dispatch(ast.children[1])
        self.emit([JUMP, start], ast)
        self._set_future(pred_fut, [JUMP_IF_FALSE_AND_POP, self._get_instructions_size()])

    def visit_apply(self, ast):
        for param in ast.children[1:]:
            self.dispatch(param)
        self.dispatch(ast.children[0])
        self.emit([CALL_FUNCTION, len(ast.children[1:])], ast)

    def visit_expr(self, ast): self.dispatch(ast.children[0])

    def visit_test(self, ast): self.dispatch(ast.children[0])

    def visit_or_test(self, ast):
        self.dispatch(ast.children[0])
        ends = []
        for expr in ast.children[1:]:
            ends.append(self._emit_future(ast))
            self.dispatch(expr)
        for end in ends:
            self._set_future(end, [JUMP_IF_TRUE_OR_POP, self._get_instructions_size()])

    def visit_and_test(self, ast):
        self.dispatch(ast.children[0])
        ends = []
        for expr in ast.children[1:]:
            ends.append(self._emit_future(ast))
            self.dispatch(expr)
        for end in ends:
            self._set_future(end, [JUMP_IF_FALSE_OR_POP, self._get_instructions_size()])

    def visit_not_test(self, ast): self._visit_keyword_expr([LOGICAL_NOT], ast)

    def visit_comparison(self, ast):
        self.dispatch(ast.children[0])
        if len(ast.children) > 1:
            self.dispatch(ast.children[2])
            self.emit([BINARY_OP[ast.children[1].additional_info]], ast)

    def _visit_bin(self, inst, ast):
        self.dispatch(ast.children[0])
        for expr in ast.children[1:]:
            self.dispatch(expr)
            self.emit(inst, ast)

    def visit_or_expr(self, ast): self._visit_bin([BINARY_OR], ast)
    def visit_xor_expr(self, ast): self._visit_bin([BINARY_XOR], ast)
    def visit_and_expr(self, ast): self._visit_bin([BINARY_AND], ast)

    def _visit_bin_op(self, ast):
        self.dispatch(ast.children[0])
        if len(ast.children) > 1:
            for index in range(1, len(ast.children) - 1, 2):
                self.dispatch(ast.children[index + 1])
                self.emit([BINARY_OP[ast.children[index].additional_info]], ast)

    def visit_shift_expr(self, ast): self._visit_bin_op(ast)
    def visit_arith_expr(self, ast): self._visit_bin_op(ast)
    def visit_term(self, ast): self._visit_bin_op(ast)

    def visit_factor(self, ast):
        if len(ast.children) == 1: self.dispatch(ast.children[0])
        else:
            self.dispatch(ast.children[1])
            if ast.children[0].additional_info == '+':
                self.emit([UNARY_POSITIVE], ast)
            elif ast.children[0].additional_info == '-':
                self.emit([UNARY_NEGATIVE], ast)
            elif ast.children[0].additional_info == '~':
                self.emit([UNARY_REVERSE], ast)

    def visit_f(self, ast):
        self.f_counter = self.f_counter + 1
        target = self.target
        f_target = '@%d' % self.f_counter
        if f_target not in self.programs[target].symbols:
            self.programs[target].symbols.append(f_target)
        params = [p.additional_info for p in ast.children if p.symbol == 'IDENTIFIER']
        program = Bytecode(instructions=[], literals=[], symbols=params,
                filename=self.target, source=self.programs[self.target].source)
        self.programs[f_target] = program
        self.target = f_target
        self.dispatch(ast.children[len(params)])
        if len(program.instructions) == 0 or program.instructions[-1][0] != RETURN_VALUE:
            self.emit([RETURN_VALUE], ast)
        self.target = target
        index = self.programs[target].symbols.index(f_target)
        self.emit([LOAD_FUNCTION, index], ast)

    def visit_IDENTIFIER(self, ast):
        index = self._get_identifier_index(ast)
        self.emit([LOAD_VARIABLE, index], ast)

    def _visit_literal(self, ast):
        self.programs[self.target].literals.append(ast.additional_info)
        self.emit([LOAD_LITERAL, len(self.programs[self.target].literals) - 1], ast)

    def visit_NULL(self, ast): self._visit_literal(ast)
    def visit_STRING(self, ast): self._visit_literal(ast)
    def visit_BOOLEAN(self, ast): self._visit_literal(ast)
    def visit_NUMBER(self, ast): self._visit_literal(ast)

    def visit_array(self, ast):
        for elem in ast.children[1:]:
            self.dispatch(elem)
        self.emit([MAKE_ARRAY, len(ast.children[1:])], ast)

    def visit_object(self, ast):
        for entry in ast.children[1:]:
            self.dispatch(entry.children[0])
            self.dispatch(entry.children[1])
        self.emit([MAKE_OBJECT, len(ast.children[1:])], ast)

    def visit_block(self, ast):
        for stmt in ast.children: self.dispatch(stmt)

    def visit_catch(self, ast):
        self.emit([CATCH_ERROR, self._get_identifier_index(ast.children[1])], ast)
        self.dispatch(ast.children[2])
        self.emit([SAVE_VARIABLE, self._get_identifier_index(ast.children[0])], ast)

class Code(object):

    def __init__(self, name, environment_frame):
        self.name = name
        self.frame = environment_frame

class Value(object):

    def __init__(self, space):
        self.space = space

class Number(Value):

    def __init__(self, space, value):
        self.space = space
        self.numbervalue = value

class Int(Number):

    type = 'int'

    def __init__(self, space, value):
        self.space = space
        self.intvalue = value

    def __repr__(self):
        return str(self.intvalue)

class BigInt(Number):

    type = 'bigint'

    def __init__(self, space, value):
        self.space = space
        self.bigintvalue = value

class Float(Number):

    type = 'float'

    def __init__(self, space, value):
        self.space = space
        self.floatvalue = value

class Str(Value):

    type = 'str'

    def __init__(self, space, value):
        self.space = space
        self.strvalue = value

class Bool(Value):

    type = 'bool'

    def __init__(self, space, value):
        self.space = space
        self.boolvalue = value

    def __repr__(self):
        return str(self.boolvalue)

class Null(Value):

    type = 'null'

class Array(Value):

    type = 'array'

    def __init__(self, space, value):
        self.space = space
        self.arrayvalue = list(value)

class Object(Value):

    type = 'object'

    def __init__(self, space, value):
        self.space = space
        self.objectvalue = {}
        for k in value:
            string_key = space.caststr(k)
            self.objectvalue[string_key] = value[k]

class Function(Value):

    type = 'function'

    def __init__(self, space, id, frame):
        self.id = id
        self.space = space
        self.frame = frame

class BuiltinFunction(Function):

    type = 'builtinfunction'

    def __init__(self, space, id):
        self.id = id
        self.space = space

class Space(object):

    def __init__(self):
        self.true = Bool(self, True)
        self.false = Bool(self, False)
        self.null = Null(self)

    def newliteral(self, s):
        if s == 'null':
            return self.null
        elif s == 'true':
            return self.true
        elif s == 'false':
            return self.false
        elif s.startswith('"'):
            return self.newstr(s[1:max(1, len(s)-1)])
        elif '.' in s or 'e' in s:
            return self.newfloat(s)
        elif len(s) > 19:
            return self.newbigint(s)
        else:
            return self.newint(s)

    def torepl(self, o): # return an rpython string
        if isinstance(o, Null):
            return 'null'
        elif isinstance(o, Bool):
            if o.boolvalue == True:
                return 'true'
            else:
                return 'false'
        elif isinstance(o, Str):
            return '"%s"' % o.strvalue
        elif isinstance(o, Int):
            return str(o.intvalue)
        elif isinstance(o, BigInt):
            return o.bigintvalue.str()
        elif isinstance(o, Float):
            return str(o.floatvalue)
        elif isinstance(o, Array):
            return '[%s]' % ','.join([
                self.torepl(e) for e in o.arrayvalue
            ])
        elif isinstance(o, Object):
            return '{%s}' % ','.join([
                '"%s":%s' % (k, self.torepl(v))
                for k, v in o.objectvalue.items()
            ])
        elif isinstance(o, Function):
            return '{"type":"function","id":"%s"}' % o.id
        else:
            raise ValueError('unknown value %s' % o)

    def toprintstr(self, o):
        if isinstance(o, Str):
            return o.strvalue
        else:
            return self.torepl(o)

    def caststr(self, o): # return rpython string
        if isinstance(o, Str):
            return o.strvalue
        else:
            return self.torepl(o)

    def castbool(self, o):
        if isinstance(o, Bool) and not o.boolvalue:
            return False
        elif isinstance(o, Null):
            return False
        elif isinstance(o, Int) and o.intvalue == 0:
            return False
        elif isinstance(o, Float) and o.floatvalue == 0.0:
            return False
        elif isinstance(o, BigInt) and o.bigintvalue.str() == '0':
            return False
        elif isinstance(o, Str) and o.strvalue == '':
            return False
        elif isinstance(o, Array) and len(o.arrayvalue) == 0:
            return False
        elif isinstance(o, Object) and len(o.objectvalue) == 0:
            return False
        return True

    def newint(self, i):
        return Int(self, int(i))

    def newfloat(self, f):
        return Float(self, float(f))

    def newbigint(self, b):
        return BigInt(self, rbigint.fromstr(b))

    def newstr(self, s):
        return Str(self, str(s))

    def newarray(self, a):
        return Array(self, a)

    def newobject(self, o):
        return Object(self, o)

    def newfunction(self, id, frame):
        return Function(self, id, frame)


def run_bin_add(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.space, left.intvalue + right.intvalue)
    if isinstance(left, Float) and isinstance(right, Int):
        return Float(left.space, left.floatvalue + right.intvalue)
    if isinstance(left, Float) and isinstance(right, Float):
        return Float(left.space, left.floatvalue + right.floatvalue)
    if isinstance(left, Int) and isinstance(right, Float):
        return Float(left.space, left.intvalue + right.floatvalue)
    else:
        raise ValueError('invalid add operation')

def run_bin_sub(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.space, left.intvalue - right.intvalue)
    if isinstance(left, Float) and isinstance(right, Int):
        return Float(left.space, left.floatvalue - right.intvalue)
    if isinstance(left, Float) and isinstance(right, Float):
        return Float(left.space, left.floatvalue - right.floatvalue)
    if isinstance(left, Int) and isinstance(right, Float):
        return Float(left.space, left.intvalue - right.floatvalue)
    else:
        raise ValueError('invalid sub operation')

def run_bin_mul(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.space, left.intvalue * right.intvalue)
    if isinstance(left, Float) and isinstance(right, Int):
        return Float(left.space, left.floatvalue * right.intvalue)
    if isinstance(left, Float) and isinstance(right, Float):
        return Float(left.space, left.floatvalue * right.floatvalue)
    if isinstance(left, Int) and isinstance(right, Float):
        return Float(left.space, left.intvalue * right.floatvalue)
    else:
        raise ValueError('invalid mul operation')

def run_bin_div(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.space, left.intvalue / right.intvalue)
    if isinstance(left, Float) and isinstance(right, Int):
        return Float(left.space, left.floatvalue / right.intvalue)
    if isinstance(left, Float) and isinstance(right, Float):
        return Float(left.space, left.floatvalue / right.floatvalue)
    if isinstance(left, Int) and isinstance(right, Float):
        return Float(left.space, left.intvalue / right.floatvalue)
    else:
        raise ValueError('invalid div operation')

def run_bin_mod(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.space, left.intvalue % right.intvalue)
    else:
        raise ValueError('invalid mod operation')

def run_bin_lshift(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.space, left.intvalue << right.intvalue)
    else:
        raise ValueError('invalid add operation')

def run_bin_rshift(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.space, left.intvalue >> right.intvalue)
    else:
        raise ValueError('invalid add operation')

def run_bin_eq(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        if left.intvalue == right.intvalue:
            return left.space.true
        else:
            return left.space.false
    elif isinstance(left, Float) and isinstance(right, Float):
        if left.floatvalue == right.floatvalue:
            return left.space.true
        else:
            return left.space.false
    elif isinstance(left, Str) and isinstance(right, Str):
        if left.strvalue == right.strvalue:
            return left.space.true
        else:
            return left.space.false
    elif isinstance(left, Bool) and isinstance(right, Bool):
        if left.boolvalue == right.boolvalue:
            return left.space.true
        else:
            return left.space.false
    elif isinstance(left, Null) and isinstance(right, Null):
        return left.space.true
    elif isinstance(left, Array) and isinstance(right, Array):
        if len(left.arrayvalue) == 0 and len(right.arrayvalue) == 0:
            return left.space.true
        if len(left.arrayvalue) != len(right.arrayvalue):
            return left.space.false
        for index in range(len(left.arrayvalue)):
            le = left.arrayvalue[index]
            re = right.arrayvalue[index]
            if not run_bin_eq(le, re).boolvalue:
                return left.space.false
        return left.space.true
    elif isinstance(left, Object) and isinstance(right, Object):
        if len(left.objectvalue) == 0 and len(right.objectvalue) == 0:
            return left.space.true
        if len(left.objectvalue) != len(right.objectvalue):
            return left.space.false
        for k in left.objectvalue:
            if k not in right.objectvalue:
                return left.space.false
            if not run_bin_eq(left.objectvalue[k], right.objectvalue[k]).boolvalue:
                return left.space.false
        return left.space.true
    else:
        return left.space.false

def run_bin_ne(left, right):
    if run_bin_eq(left, right).boolvalue:
        return left.space.false
    else:
        return left.space.true

def run_bin_gt(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.space, left.intvalue > right.intvalue)
    if isinstance(left, Float) and isinstance(right, Int):
        return Float(left.space, left.floatvalue > right.intvalue)
    if isinstance(left, Float) and isinstance(right, Float):
        return Float(left.space, left.floatvalue > right.floatvalue)
    if isinstance(left, Int) and isinstance(right, Float):
        return Float(left.space, left.intvalue > right.floatvalue)
    else:
        raise ValueError('invalid ge operation')

def run_bin_ge(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.space, left.intvalue >= right.intvalue)
    if isinstance(left, Float) and isinstance(right, Int):
        return Float(left.space, left.floatvalue >= right.intvalue)
    if isinstance(left, Float) and isinstance(right, Float):
        return Float(left.space, left.floatvalue >= right.floatvalue)
    if isinstance(left, Int) and isinstance(right, Float):
        return Float(left.space, left.intvalue >= right.floatvalue)
    else:
        raise ValueError('invalid ge operation')

def run_bin_lt(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.space, left.intvalue < right.intvalue)
    if isinstance(left, Float) and isinstance(right, Int):
        return Float(left.space, left.floatvalue < right.intvalue)
    if isinstance(left, Float) and isinstance(right, Float):
        return Float(left.space, left.floatvalue < right.floatvalue)
    if isinstance(left, Int) and isinstance(right, Float):
        return Float(left.space, left.intvalue < right.floatvalue)
    else:
        raise ValueError('invalid ge operation')

def run_bin_le(left, right):
    if isinstance(left, Int) and isinstance(right, Int):
        return Int(left.space, left.intvalue <= right.intvalue)
    if isinstance(left, Float) and isinstance(right, Int):
        return Float(left.space, left.floatvalue <= right.intvalue)
    if isinstance(left, Float) and isinstance(right, Float):
        return Float(left.space, left.floatvalue <= right.floatvalue)
    if isinstance(left, Int) and isinstance(right, Float):
        return Float(left.space, left.intvalue <= right.floatvalue)
    else:
        raise ValueError('invalid ge operation')

class CodeContext(object):

    def __init__(self, name, pc, bytecode, tos, interpreter):
        self.name = name
        self.pc = pc
        self.bytecode = bytecode
        self.op = bytecode.instructions[pc]
        self.opcode = self.op[0]
        self.opval = 0 if len(self.op) == 1 else self.op[1]
        self.opname = CODES[self.opcode]
        self.tos = tos
        tos.save_pc(pc)
        self.interpreter = interpreter

def make_dispatch_function(**dispatch_table):
    def dispatch(self, ctx):
        func = dispatch_table[ctx.opname]
        return func(self, ctx)
    return dispatch

class CreateOpcodeDictionaryMetaclass(type):

    def __new__(cls, name_, bases, dct):
        dispatch_table = {}
        for name, value in dct.iteritems():
            if name.startswith("run_"):
                dispatch_table[name[len("run_"):]] = value
        dct["dispatch"] = make_dispatch_function(**dispatch_table)
        return type.__new__(cls, name_, bases, dct)

class BaseInterpreter(object):
    __metaclass__ = CreateOpcodeDictionaryMetaclass

class Interpreter(BaseInterpreter):

    def __init__(self, entry, program):
        self.running = True
        self.exit_code = 0
        self.program = program
        self.error = None
        self.space = Space()
        self.stack = [Frame(entry, space=self.space, bytecode=program.programs[entry])]

    def run_PRINT(self, ctx):
        val = ctx.tos.pop()
        print(self.space.toprintstr(val))
        ctx.pc += 1

    def run_LOAD_LITERAL(self, ctx):
        lit = ctx.bytecode.get_literal(ctx.opval)
        val = self.space.newliteral(lit)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_SAVE_VARIABLE(self, ctx):
        sym = ctx.bytecode.get_symbol(ctx.opval)
        ctx.tos.save(sym, ctx.tos.pop(), ctx.tos)
        ctx.pc += 1

    def run_LOAD_VARIABLE(self, ctx):
        sym = ctx.bytecode.get_symbol(ctx.opval)
        ctx.tos.push(ctx.tos.load(sym))
        ctx.pc += 1

    def run_MAKE_ARRAY(self, ctx):
        length = ctx.opval
        args = [ctx.tos.pop() for _ in range(length)]
        val = self.space.newarray(args)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_MAKE_OBJECT(self, ctx):
        length = ctx.opval
        obj = {}
        for _ in range(ctx.opval):
            key = ctx.tos.pop()
            value = ctx.tos.pop()
            assert isinstance(value, Value)
            obj[key] = value
        val = self.space.newobject(obj)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_LOAD_FUNCTION(self, ctx):
        sym = ctx.bytecode.get_symbol(ctx.opval)
        val = self.space.newfunction(sym, ctx.tos)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_CALL_FUNCTION(self, ctx):
        func_val = ctx.tos.pop()
        if func_val.id.startswith('@@'):
            params = []
            for i in range(ctx.opval):
                params.append(ctx.tos.pop())
            builtin = MODULE[func_val.id[2:]]
            builtin(BuiltinContext(self, ctx.tos, ctx.bytecode, params))
            ctx.pc += 1
        else:
            func_bytecode = self.program.programs[func_val.id]
            parent_frame = func_val.frame
            new_frame = Frame(name=func_val.id, pc=0,
                    space=self.space,
                    parent=parent_frame, bytecode=func_bytecode)
            for i in range(ctx.opval):
                argv_i_sym = func_bytecode.get_symbol(ctx.opval - i - 1)
                new_frame.save(argv_i_sym, ctx.tos.pop(), new_frame)
            self.stack.append(new_frame)
            ctx.name = func_val.id
            ctx.pc = 0

    def run_RETURN_VALUE(self, ctx):
        ret_val = ctx.tos.pop() if ctx.tos.evaluations else self.space.null
        self.stack.pop()
        prev_tos = self.stack[-1]
        prev_tos.push(ret_val)
        ctx.name = prev_tos.name
        ctx.pc = prev_tos.pc + 1

    def run_JUMP(self, ctx):
        ctx.pc = ctx.opval

    def run_JUMP_IF_TRUE_AND_POP(self, ctx):
        val = ctx.tos.pop()
        ctx.pc = ctx.opval if self.space.castbool(val) else ctx.pc + 1

    def run_JUMP_IF_FALSE_AND_POP(self, ctx):
        val = ctx.tos.pop()
        ctx.pc = ctx.opval if not self.space.castbool(val) else ctx.pc + 1

    def run_JUMP_IF_TRUE_OR_POP(self, ctx):
        val = ctx.tos.peek()
        if self.space.castbool(val):
            ctx.pc = ctx.opval
        else:
            ctx.tos.pop()
            ctx.pc += 1

    def run_JUMP_IF_FALSE_OR_POP(self, ctx):
        val = ctx.tos.peek()
        if self.space.castbool(val):
            ctx.tos.pop()
            ctx.pc += 1
        else:
            ctx.pc = ctx.opval

    def run_EXIT(self, ctx):
        self.exit_code = ctx.opval
        self.running = False

    def run_BINARY_ADD(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_add(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_BINARY_SUB(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_sub(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_BINARY_MUL(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_mul(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_BINARY_DIV(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_div(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_BINARY_MOD(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_mod(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_BINARY_LSHIFT(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_lshift(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_BINARY_RSHIFT(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_rshift(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_BINARY_EQ(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_eq(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_BINARY_NE(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_ne(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_BINARY_GE(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_ge(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_BINARY_LE(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_le(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_BINARY_GT(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_gt(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_BINARY_LT(self, ctx):
        right, left = ctx.tos.pop(), ctx.tos.pop()
        val = run_bin_lt(left, right)
        ctx.tos.push(val)
        ctx.pc += 1

    def run_LOGICAL_NOT(self, ctx):
        val = ctx.tos.pop()
        val = self.space.castbool(val)
        val = self.space.false if val else self.space.true
        ctx.tos.push(val)
        ctx.pc += 1

    def run_CATCH_ERROR(self, ctx):
        sym = ctx.bytecode.get_symbol(ctx.opval)
        ctx.tos.catch(sym)
        ctx.pc += 1


class Frame(object):
    def __init__(self, name, space, pc=0, parent=None, bytecode=None):
        self.bindings = {}
        self.space = space
        self.evaluations = []
        self.codes = {}
        self.parent = parent
        self.pc = pc
        self.name = name
        self.bytecode = bytecode
        self.error_var = None

    def save_pc(self, pc):
        self.pc = pc

    def save(self, key, value, frame=None):
        self.bindings[key] = value
        if self.error_var is not None:
            self.error_var = None

    def load(self, key):
        if key in self.bindings:
            return self.bindings[key]
        elif self.parent is not None:
            return self.parent.load(key)
        elif key in MODULE:
            return Function(self.space, '@@' + key, None)
        else:
            raise Exception('unknown variable: %s' % key)

    def catch(self, error):
        self.error_var = error

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
    def __init__(self, machine, tos, bytecode, params):
        self.machine = machine
        self.tos = tos
        self.bytecode = bytecode
        self.params = params

def crash_stack(machine):
    locations = []
    while len(machine.stack) > 0:
        tos = machine.stack[-1]
        if machine.error is not None and tos.error_var is not None:
            tos.push(machine.space.null)
            tos.save(tos.error_var, machine.error)
            machine.error = None
            return tos
        else:
            locations.append(get_location(tos.pc, tos.name, machine.program))
            machine.stack.pop()
    print('Traceback: %s' % machine.space.toprintstr(machine.error))
    for loc in locations:
        print(loc)
    if machine.exit_code == 0:
        machine.exit_code = 1

def mainloop(filename, source):
    pc = 0
    name = filename
    compiler = Compiler(name)
    compiler.compile(name, source)
    if name not in compiler.programs:
        return 1
    interpreter = Interpreter(name, compiler)
    while pc < len(compiler.programs[name].instructions) and interpreter.running:
        jitdriver.jit_merge_point(pc=pc, name=name, program=compiler, machine=interpreter)
        tos = interpreter.stack[-1]
        bytecode = interpreter.program.programs[name]
        ctx = CodeContext(name, pc, bytecode, tos, interpreter)
        interpreter.dispatch(ctx)
        name, pc = ctx.name, ctx.pc
        if interpreter.error is not None:
            tos = crash_stack(interpreter)
            if tos is None:
                break
            else:
                name = tos.name
                pc = tos.pc + 1
    return interpreter.exit_code

def run(filename):
    program_contents = ""
    try:
        fp = os.open(filename, os.O_RDONLY, 0777)
    except OSError as e:
        print str(e)
        return 1
    while True:
        read = os.read(fp, 4096)
        if len(read) == 0: break
        program_contents += read
    os.close(fp)
    return mainloop(filename, program_contents)

def entry_point(argv):
    try:
        filename = argv[1]
    except IndexError:
        print "You must supply a filename"
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
