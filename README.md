# AO

AO, pronouncing as A-O, is a tiny interpreter language.

It supports:

* JSON as primitive types: `null`, `true`, `false`, `0`, `1.0`, `"string"`, `[1, 2]`, `{"key": "value"}`.
* Assignment: `x = 1;`.
* Function: `double = f (a) { return a * a; };`.
* Arithmetic: `+`, `-`, `*`, `/`, `%`, `<<`, `>>`, `&`, `|`, `~`.
* Comparison: `==`, `!=`, `>`, `>=`, `<`, `<=`.
* Logical: `and`, `or`, `not`.
* Error handling: `value or error = a_function_might_raise_error();`

## Getting Started

Due to its early stage, it's not appropriate to publish a release.
If you want to get an AO executable, please read `Develop` to compile interpreter for now.

## A Glance of AO

All builtin types are JSON based values, expect Function.
No class, no struct, no trait, no interface, no ..., etc.

```
a = 1;
b = 1.0;
c = null;
d = true;
e = false;
f = "hello world";
g = [1, 2, 3];
h = {"key": "value"};
```

Control flow is very traditional.

```
a = 1;
b = 2;
c = 3;

if (a <= b and b <= c) {
    print "a, b, c are increasing.";
} elif (a >= b and b >= c) {
    print "a, b, c are decreasing."
} else {
    print "a, b, c are in random order."
}

while (c <= 0) {
    print c;
    c = c - 1;
}

while (true) {
    print "loop";
}
```

Defining function is just yet another assignment.

```
i = f (j) {
    return j * 2;
};

j = i(24);
print j;
```

Function can be closure.

```
g = f (m) {
    return f(n) {
        return m + n;
    };
};

plus42 = f(42);
print plus42(42);
```

Catch errors.

```
badfunction = f () {
    raise("It can be any type, str, int, object, etc.");
    # It never returns 1 here.
    return 1;
};

value or error = badfunction();
```

## Develop

To build `ao` binary, you will need to install Python 2.7, virtualenv and rpython:

```
$ virtualenv venv
$ source venv/bin/activate
$ pip install rpython

# disable jit.
$ rpython targetao.py

# enable jit.
$ rpython -Ojit targetao.py
```

TODO:

* Document of language syntax.
* Implement builtin functions.
* Implement standard libraries.
* Implement system programming libraries.

## Contribute

The AO Philosophy:

* Simplicity.
    * AO aims to provie simple syntax so that new learners can quickly start learning programming.
    * No matter what a mess the implementation is, keep the interface simple.
* Correctness.
    * Make it right. Fix the bug if we catch one. :)
* Battery-included.
    * Softwares are meant to solve problems. AO ships with a set of handy libraries including text processing, binary data processing, time processing, a set of advanced data structures and algorithms, mathmatical calculation, functional programming, system programming, serialization and deserialization, protocols and formats, logging, web programming, game programming, concurrency, IPC and networking, parsers for different languages, multimedia programming, i18n, GUI programming, testing and debugging packaging and distribution, clients for data-intensive systems, science programming, shell programming, etc.

## License

AO comes with MIT license.
