# Language

AO aims to provide simple syntax. This page defines AO syntax.

## Int Literals

| Examples      | Description |
| ------------- | ----------- |
| 0, -42, 100   | Evaluation of a number literal yields a numeric value. |

## Float Literals

| Examples            | Description |
| ------------------- | ----------- |
| 0.0, -42.0, 100e1   | Evaluation of a number literal yields a numeric value. |

## Bool Literals

| Examples    | Description |
| ----------- | ----------- |
| true, false | Evaluation of a bool literal yields a boolean value. It's either true or false. |

## Null Literal

| Examples    | Description |
| ----------- | ----------- |
| null        | Evaluation of a null literal yields an empty value. |

## Str Literals

| Examples             | Description |
| -------------------- | ----------- |
| "wrap it with quote" | Evaluation of a str literal yields a string value. |
| "use \" to escape"   | `\"` can help you use `"` in string. |
| "use \\ to escape"   | `\\` can help you use `\` in string. |

## Array Literals

| Examples         | Description |
| ---------------- | ----------- |
| [], [1, 2, 3]    | Evaluation of an array literal yields an array of elements. Each element is one of the other literals. |
| [[1, 2], [2, 3]] | Arrays can be nested. |

## Object Literals

| Examples             | Description |
| -------------------- | ----------- |
| {}, {"key": "value"} | Evaluation of an object literal yields a key-value mapping value. Key must be a string. Value can be any literals. |

## Variables

Use `=` to bind a value to a name:

```
[1]: a = 1;
```

Values can be value to another name:

```
[1]: a = 1;

[2]: b = a;
```

Values can be used in array and object:

```
[1]: a = 1;

[2]: b = {"a": a};

[3]: c = [a, b];
```

## Comparisons

Comparison operators include `==`, `!=`, `>`, `>=`, `<`, `<=`.

```
[1]: 1 == 1
true

[2]: 1 == 2
false

[3]: 1 != 2
true

[4]: 2 != 2
false

[5]: 1 > 2
false

[6]: 1 < 2
true

[7]: 1 >= 1
true

[8]: 1 <= 1
true
```

## Logical Tests

Logical tests include `and`, `or`, and `not`.

```
[1]: 1 == 1 and 1 == 2
false

[2]: 1 == 2 or 1 == 1
true

[3]: not 1 == 2
true
```

Operator `and` is short-curcuited. Once a condition is evaluated as false, then the rest of condition won't be evaluated:

Operator `or` is short-curcuited as well. Once a condition is evaluated as true, then the rest of condition won't be evaluated.

## Arithmetic Operators

Arithmetic operators include `+`, `-`, `*`, `/`, `%`, `<<`, `>>`, `&`, `|`, `~`.

All operators can be applied to int and float literals.

```
[1]: 1 + 1
2

[2]: 1 - 1
0

[3]: 2 * 2
4

[4]: 4 / 2
2

[5]: 3 % 2
1

[6]: 1 << 2
4

[7]: 4 >> 2
1

[8]: 42 & 1
0

[9]: 42 | 1
43

[10]: ~42
-43
```

## Function

Defining function is just yet another assignment.

```
[1]: i = f (j) {
         return j * 2;
     };

[2]: j = i(24);

[3]: print j;
48
```

Functions can be nested and as return value.

```
g = f (m) {
    return f(n) {
        return m + n;
    };
};

plus42 = g(42);
print plus42(42);
```

## Operator reload

Some operators can be applied to strings: `+`, `-`, `*`, `/`. The operation yields a new value instead of modification in-place.

```
[1]: "hello" + ", " + "world"
"hello, world"

[2]: "hello, world" - "o"
"hell, wrld"

[3]: "hello" * 3
"hellohellohello"

[4]: "hellohellohello" / 3
"hello"

[5]: "hello hello hello" / " "
["hello", "hello", "hello"]
```

Some operators can be applied to array: `<<`, `>>`, `+`, `-`. The operation yields a new value instead of modification in-place.

```
[1]: [1, 2, 3] << 4
[1, 2, 3, 4]

[2]: 0 >> [1, 2, 3, 4]
[0, 1, 2, 3, 4]

[3]: [1, 2, 3] << [1, 2]
[1, 2, 3, [1, 2]]

[4]: [1, 2] >> [1, 2, 3]
[[1, 2], 1, 2, 3]

[5]: [1, 2, 3] + [1, 2]
[1, 2, 3, 1, 2]

[6]: [1, 2, 3, 1, 2] - [1, 2]
[3, 1, 2]
```

For array, it also allow `&` for mapping values to a function, allow `|` for filtering values from a function:

```
[1]: double = f (n) { return n + n; };

[2]: [1, 2, 3] & double
[2, 4, 6]

[3]: is_even = f (n) { return n % 2 == 0; };

[4]: [1, 2, 3, 4] | is_even
[2, 4]
```

## Builtin Functions

`type(value)` return string of type name:

```
[1]: type(1)
"int"

[2]: type(1.0)
"float"

[3]: type(true)
"bool"

[4]: type(null)
"null"

[5]: type([])
"array"

[6]: type({})
"object"

[7]: type("hello world")
"string"

[8]: type(f () { return null; })
"function"
```
