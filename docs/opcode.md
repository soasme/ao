# Opcode

Opcode is a number representing what action needs to be done in interpreter.

## `LOAD_LITERAL`

Literals are JSON constant values and stored in VM. All processes can load literals from VM.

## `LOAD_VARIABLE`

Variables are key-value bindings and stored in Actor stack frame. Variables can't be loaded outside of the Actor.

## `LOAD_BUILTIN`

Builtins are VM implemented functions. All I/O related builtin functions are executing asynchronously. All builtin function

## `LOAD_CLOSURE`

Instruction `LOAD_CLOSURE` loads a sequence of bytecode into top stack frame as closure. The closure has a reference to outer environment.

## `ASSIGN`

Instruction `ASSIGN` creates key-value bindings for designated elements in top stack frame.

## `MAKE_FUNCTION`

Instruction `MAKE_FUNCTION` turns a closure into a function.

## `CALL_FUNCTION`

Instruction `CALL_FUNCTION` create a new stack frame for function execution.

## `RETURN_VALUE`

Instruction `RETURN_VALUE` returns top of stack to the caller of function, which is the second element in the stack actually.

## `ARITH_OP`

Instruction `ARITH_OP` pops designated values from stack, does some math calculation, and then pushes back to stack.

## `LOGIC_OP`

Instruction `LOGIC_OP` pops designated values from stack, does some logical calculation, and then pushes back to stack.

## `PRINT`

Instruction `PRINT` output values into system stdout.

## `MAKE_OBJECT`

Instruction `MAKE_OBJECT` creates an object based on designated elements in the stack.

## `MAKE_ARRAY`

Instruction `MAKE_ARRAY` creates an array based on designated elements in the stack.
