Mutagen Programming Language
============================

The Mutagen programming language is a simple, readable purely functional language. Its syntax is very much influenced by Python.

Mutagen is based on a few concepts:
* Compilers should be smart, and most of the "grunt" work of traditional imperative programming should not be forced on humans
* Simplicity in language design pays off many-fold in creating a more powerful compiler
* The power of functional programming languages should not require use of hard-to-read, super-technical, full-of-notation languages
* Metaprogramming becomes an absolutely essential feature for large yet flexible codebases

Mutagen tries to take the pleasure of writing and reading Python programs and extend it to "mathematically pure" programming, where the compiler can much more easily reason about the programs (which is very difficult for Python).

One of the main goals is to create the fastest-performing programming language for virtually any task. This is a hard problem, and we're running against orders of magnitude more effort expenditure in compilers for traditional imperative languages (and even traditional functional languages).

So how might Mutagen achieve such speed goals? There's several possible reasons:
* Language designed from the beginning to allow for fast execution. All language constructs ultimately reduce to representations in machine words, blocks of memory, and generic arithmetic instructions in a natural way.
* Integrated tuning facilities in the language. Higher-level algorithmic information available to the programmer should be communicable to the compiler in some way, so that the compiler can make both high-level and low-level decisions, and reason about the interactions between them.
* A compiler written entirely in Mutagen, which allows the same style of language-integrated tuning to improve the compiler itself (both in compilation time and generated code).
* Unbounded compilation times, to allow for more expensive search-based compiler algorithms (ideally ones that scale down on the low end as well, for fast minimum compilation times).
* Extensive caching in the compiler implementation, made possible by referential transparency, enabling the benefits of long compilation times on changing-but-still-similar codebases without the long latency.
* Better engineered stack. This remains to be seen, of course, but the hope is that removing years of cruft from other languages will allow Mutagen to have a better rate of improvement in general.
* In the future, perhaps integrated microarchitectural simulations for hardware-feedback assisted tuning, or eventually fully integrated hardware/software codesign.

With readability, smart compilers, and superfast performance as fundamental language design choices, the overall goal of Mutagen is to increase programmer/computer efficiency exponentially, eventually leading to a blissful Utopian society free of want.

Current State
-------------

Mutagen is still in a very prototypical state. It is almost entirely implemented as a Python bootstrap program, using PLY to support its lexing/parsing needs. The goal is to keep the bootstrap as small as possible, while allowing for the duplication of the bootstrap, and later compilation stages, in Mutagen, so as to allow for Pythonless execution. The bootstrap is a very simple interpreter, and actually keeps considerable state in the form of symbol tables, Python-backed data structures, and imperative-style I/O. This is just for simplicity in the initial implementation, and the language itself has no mutable state (or at least it will, once it has some monad-like facility).

As of now, only the lexer has an equivalent implementation in Mutagen. It is quite reasonable in terms of code size and readability when compared to the Python version, but its speed is still quite lacking due to the slow interpretive regex engine, as well as the double interpreter layer (Mutagen on top of Python).

Syntax
------

Mutagen's syntax is very similar to Python. As of now, there are a few small changes:

* Classes are "case classes" rather than relying on `__init__()`. `__init__()` is still available, and will eventually only work as a "pre-processing" step before calling the base constructor. As of now, since there is no polymorphism, the presence of `__init__()` overrides the base constructor, and must return a dictionary of attributes and values. This special constructor is also different from Python in that it does not accept a `self` parameter (as the object has not been created yet, and it would be immutable anyhow). Examples:
```python
    class Example(attr1, attr2):
        # A basic constructor is defined by default, so the attributes are
        # available in methods:
        def sum(self):
            return self.attr1 + self.attr2

    class ExampleWithInit(ignored_attr):
        # As of now, any parameters to a class with __init__ are ignored, so
        # the object must be constructed manually
        def __init__(attr1, attr2):
            return {'attr1': attr1, 'attr2': attr2}
```
* Function and class parameters have an optional type attribute, specified with a colon and expression after the parameter name. This allows type checking, and will eventually allow polymorphism and type inference. Examples:
```python
    def add_ints(a: int, b:int):
        return a + b

    class Point(x: int, y: int):
        def distance_from_origin(self):
            return add_ints(self.x * self.x, self.y * self.y)
```
* Braces can also be used to delimit blocks, interchangeably with indentation-based blocks as in Python. To use brace-delimited blocks, semicolons must be explicity used between statements (since newlines are ignored inside braces, as in Python), and the trailing colon beginning the block must be omitted. For example, these blocks are equivalent:
```python
    if x == y:
        x = 0
        y = 1

    if x == y { x = 0; y = 1; }
```
* Imports have a special syntactic form for allowing relative imports:
```python
    import module from 'path/to/module.mg'
```
* `None` in Python has been replaced by `Nil` in Mutagen.
* There are no tuples, since lists are already immutable. Thus, there is no parenthesized tuple syntax like `(a, b)`, or implicit tuple syntax like `a, b`.
* `lambda` is just a synonym for a `def` without a function identifier, not a limited expression as in Python. Thus, if used in an expression, it will usually require braces instead of indentation, semicolons as statement delimiters, and an explicit return statement. Example:
```python
    lambda(x) {return x + 1;}
```
