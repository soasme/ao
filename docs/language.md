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
| [[1, 2], [2, 3]] | Lists can be nested. |
