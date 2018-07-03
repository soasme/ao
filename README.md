# AO

AO, pronouncing as A-O, is a tiny interpreter language.
Its runtime gets implemented in only 500+ lines of code.

It supports:

* JSON types: `null`, `true`, `false`, `0`, `1.0`, `"string"`, `[1, 2]`, `{"key": "value"}`.
* Assignment: `x = 1;`.
* Function: `double = f (a) { return a * a; };`.
* Arithmetic: `+`, `-`, `*`, `/`, `%`, `<<`, `>>`, `&`, `|`, `~`.
* Comparison: `==`, `!=`, `>`, `>=`, `<`, `<=`.

## Getting Started

Due to its early stage, it's not appropriate to publish a release.
If you want to get an AO executable, please read `Develop` to compile interpreter for now.

## A Glance of AO

All builtin types are JSON based values, plus Lambda.
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

## Develop

To build `ao` binary, you will need to install Python 2.7 and rpython:

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
    * Softwares are meant to solve some problem. AO ships with a set of handy libraries.

## License

AO comes with MIT license.
