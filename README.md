Mutagen Programming Language
============================

The Mutagen programming language is a simple, readable purely functional language. Its syntax is very much influenced by Python, and will eventually, if all goes according to plan, be extremely fast.

Mutagen is based on a few concepts:
* Compilers should be smart, and most of the "grunt" work of traditional imperative programming should not be forced on humans
* Simplicity in language design pays off many-fold in creating a more powerful compiler
* The power of functional programming languages should not require the use of hard-to-read, super-technical, full-of-notation languages
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

#### Some quick notes about this project and its author
Mutagen is definitely an ambitious project. I am well aware I sound a little crazy... and well, I kinda am. A lot of this stuff is fuzzy, and might not turn out as well as I imagine. My knowledge of a lot of the deep theoretical computer science-y aspects of some languages (particularly ones like Haskell) could be better--I'm mostly self-taught in these areas.

That said, I am very sure of the need for better programming languages, and their potential. I'm fairly confident in my ability to (eventually) deliver something in this area, and my vision for what Mutagen can become is definitely a lot more detailed in my head than what you'll find written in this repository.

To convince you I'm not completely full of shit, I do have a bit of credibility to my name:
* I spent many of my younger years in the computer chess world, and I do have a World Computer Chess Co-Championship title to my name (which might someday turn into a sole championship). That's actually not as impressive as it sounds... it's a long crazy story, but if you're bored/masochistic, read starting from [here](http://chessprogramming.wikispaces.com/Rybka#Controversies). I bring it up since I learned quite a bit about software engineering from my time in computer chess--many lessons that haven't percolated to most of the software world at large (or even to a lot of computer chess people for that matter). And interestingly, there's a lot more overlap between a chess-playing program and a compiler than you'd think at first.
* I worked on a fairly similar language (Mash), targeted at (the-microarchitecture-formerly-known-as-)[Larrabee](https://en.wikipedia.org/wiki/Larrabee_%28microarchitecture%29), when I was employed at Intel around 2010-2011. (Cheers to Matt Craighead, who initially created it; our discussions inspired a lot of Mutagen). It never got too far off the ground before we left Intel, but basically it was coming from the opposite direction--essentially from assembly towards Python, rather than from Python towards assembly. There's many differences, and Mutagen is trying to be more pure/mathematical from the start, but generally the end goal was pretty much the same--a superfast, functional, Pythonic language.

Mutagen will be, I hope, the foundation for all of the programming I want to do in the future. I like things to go fast, and I want to tinker with algorithms/optimization with as little friction as possible. If you think I might be on to something, I'd love to talk/collaborate, especially if you have background in functional programming/compilers/optimization/microarchitecture. You should probably get in touch before writing any nontrivial patches, but really, I do want to hear from you.

Current State
-------------

Mutagen is still in a very prototypical state. It is almost entirely implemented as a Python bootstrap program. The goal is to keep the bootstrap as small as possible, while allowing for the duplication of the bootstrap, and later compilation stages, in Mutagen, so as to allow for Pythonless execution. The bootstrap is a very simple interpreter, and actually keeps considerable state in the form of symbol tables, Python-backed data structures, and imperative-style I/O. This is just for simplicity in the initial implementation, and the language itself has no mutable state (or at least it will, once it has some monad-like facility for interfacing with the world, particularly I/O).

As of now, the Mutagen self-hosting compiler has a lexer, most of a parser, a basic assembler, and a very bare-bones register allocator. It is quite reasonable in terms of code size and readability, but its speed is still very much lacking due to the double interpreter layer (Mutagen on top of Python), and many non-optimal aspects of the current implementation (in the name of simplicity/laziness).

Semantics
---------

So what does a "functional python" really mean? Mutagen is fundamentally based on purity, that is, immutability of every object. But given the Python influence, every variable is really a reference. So as opposed to most other functional languages, variables can be reassigned--it's the values that the variables point to which are immutable.

```python
x = 0
x = [1, 2, 3]   # Sure thing!
x[0] = 5        # This won't work--the value of x, a list, is immutable
```

This choice comes with a cost. For one, basically every existing computer works in a very imperative/mutable way, and the compiler will have to do considerable work for purely functional representations of data to be performant. Also, there is the syntactic issue of describing updates of state in a coherent way. Finally, a functional language cannot go very far without recursion, and given this data model, objects cannot directly reference themselves, so there is no direct way for recursive functions or types to exist.

For now, Mutagen is punting on all of these issues, with the hope that the answers will solidify as the language evolves. For the problem of recursion, a [fixed-point combinator](https://en.wikipedia.org/wiki/Fixed-point_combinator) can be used as a workaround for now.

Syntax
------

Mutagen's syntax is very similar to Python. There are some small changes though:

* To deal with syntactic description of state updates, there is a basic modification operator, the left arrow (`<-`), which allows functionality similar to `__setattr__`/`__setitem__` in Python. Since Mutagen doesn't allow this common type of Python statement:
```python
some_list[x] = 0
list_of_lists[x][y] = 1
list_of_objects[x].attr = 'hello'
```
...and the obvious way of doing this with immutable objects gets icky really fast:
```python
some_list = some_list[:x] + [0] + some_list[x + 1:]
list_of_lists = list_of_lists[:x] + [list_of_lists[x][:y] + [1] + list_of_lists[x][y + 1:]] + list_of_lists[x + 1:]
list_of_objects = list_of_objects[:x] + [SomeObject(list_of_objects[x].some_attr,
        list_of_objects[x].and_another_attr,
        list_of_objects[x].maybe_another_one_too_why_not_I_could_do_this_all_day,
        'hello') # <-- The only attribute we are actually trying to change
    ] + list_of_objects[x + 1:]
```
...the following should be used to preserve sanity:
```python
some_list = some_list <- [x] = 0
list_of_lists = list_of_lists <- [x][y] = 1
list_of_objects = list_of_objects <- [x].attr = 'hello'
```
And for even more fun, these can be chained with commas:
```python
some_dict = some_dict <- ['a'] = 1, ['b'] = 2, ['c'] = 3
```
* Classes are "case classes" rather than relying on `__init__()`. `__init__()` is still available, and will eventually only work as a "pre-processing" step before calling the base constructor. As of now, since there is no polymorphism, the presence of `__init__()` overrides the base constructor, and must return a dictionary of attributes and values. This special constructor is also different from Python in that it does not accept a `self` parameter (as the object has not been created yet, and it would be immutable anyhow). Examples:

```python
class Example(attr1, attr2):
    # A basic constructor is defined by default, so the attributes are
    # available in methods:
    def sum(self):
        return self.attr1 + self.attr2

class ExampleWithInit(ignored_attr):
    # As of now, any parameters to a class with __init__ are ignored, so
    # the object must be constructed manually. Of course, this is horrible, but
    # the language is too immature to replace it yet.
    def __init__(attr1, attr2):
        return {'attr1': attr1, 'attr2': attr2}
```
* Type annotations are used quite similarly to Python annotations, but are actually enforced. Function and class parameters have an optional type attribute, specified with a colon and expression after the parameter name. Additionally, function return types can be specified with a right arrow (`->`) after the parameters and before the colon. This allows type checking, and will eventually allow polymorphism and other neat stuff. Examples:

```python
def add_ints(a: int, b: int) -> int:
    return a + b

class Point(x: int, y: int):
    def manhattan_distance_from_origin(self) -> int:
        return add_ints(self.x, self.y)
```
* **NOTE: this feature is currently disabled, since it's not really needed at the moment. It will eventually come back when it can be implemented cleanly.** To support basic algebraic types (which will be the basis of all of Mutagen's type system once all 'builtin' classes are unified with user-defined classes), 'tagged' unions (a.k.a. sum types) are supported. These take parameters like a case class, but each parameter defines a distinct value that the union can take. Without a type for the parameter, it defines a "unit" type--a value that holds no data and only has one possible value. This can be used to create enums. With a type, the parameter can hold arbitrary data. Examples:

```python
union Bool(false, true):
    # Demonstrating the use of methods on unions
    def __not__(self):
        # Enums do not have any sort of int-like behavior.
        return {Bool.false(): Bool.true(), Bool.true(): Bool.false()}[self]

# A la Haskell's Maybe monad
union Maybe(Nothing, Just: class(value)):
    pass
nothing = Maybe.Nothing()
just = Maybe.Just(7)
```
* Imports have a special syntactic form for allowing relative imports:

```python
import module from 'path/to/module.mg'
```
This might change in the future, as the module/packaging system gets fleshed out.
* There are no tuples, since lists are already immutable. Thus, there is no parenthesized tuple syntax like `(a, b)`, or implicit tuple syntax like `a, b`.
* Relatedly, destructuring assignment/for loop targets are supported, but they also must use explicit list notation rather than implicit tuple notation:

```python
[a, [b, [c]]] = nested_data
for [i, x] in enumerate(data):
    pass
```
* Dictionary iteration iterates over key-value pairs instead of keys as in Python. This might change in the future if any particularly compelling rationale for key-only iteration arises.

```python
d = {'a': 1, 'b': 2}
for [k, v] in d:
    print(k, v)

# Outputs:
# a 1
# b 2
```
* `lambda` is very slightly modified, in that the lambda parameters have parentheses around them. This is to keep the grammar simple, as well as to allow lambda parameters to have type annotations.

```python
plus_one = lambda(x: int): x + 1
```
* To deal with the 'lambda problem' from Python (that is, `lambda` only allows a single expression and not any statements), a special form of `def` is available. When `def` is used without a function identifier, the entire def is treated as an expression. In this case, it will usually require braces instead of indentation, semicolons as statement delimiters, and an explicit return statement (see the next item for more on this). A simple example:

```python
check_x = def(x) {
    assert x > 0;
}
```
* Braces can also be used to delimit blocks, interchangeably with indentation-based blocks as in Python. To use brace-delimited blocks, semicolons must be used to delimit statements (since newlines/indentation are ignored inside braces, as in Python), and the trailing colon beginning the block must be omitted. Now, before the Python community beheads me for even considering this blasphemous proposal, I will note that indentation-based blocks are still greatly preferred, and should be considered idiomatic. Brace-delimited blocks generally are only useful in cases such as lists/dictionaries of functions, where `lambda` is insufficient, and indentation-based blocks are not possible. To demonstrate this new brace syntax, these example blocks are equivalent:

```python
if x == y:
    x = 0
    y = 1

if x == y { x = 0; y = 1; }
```
And here's a (super-contrived) example of something that's not easily possible in Python:

```python
list_of_functions_that_have_real_statements_in_them = [
    def (x) { assert x > 0; },
    def (x) { yield x; yield x * 2; }
]
```
* Closures are supported, but the captured values are static. That is, whenever the inner function definition is evaluated, the values of variables in parent scopes are bound to the function at that point. This makes metaprogramming somewhat simpler, and since any benefits of dynamic scoping would require mutable variables, there is little downside given a purely functional environment. An example of this behavior:

```python
lambdas = [lambda: x for x in [0, 1]]
for l in lambdas:
    print(l())

# Outputs:
# 0
# 1
```
