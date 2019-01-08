Mutagen Programming Language
============================

The Mutagen programming language is a simple, readable purely functional language. Its syntax is very much influenced by Python, and will eventually, if all goes according to plan, be extremely fast.

Mutagen is based on a few concepts:
* Compilers should be smart, and most of the "grunt" work of optimizing traditional imperative programming should not be forced on humans
* Simplicity in language design pays off many-fold in creating a more powerful compiler
* The power of functional programming languages should not require the use of hard-to-read, super-technical, full-of-notation languages
* Metaprogramming becomes an absolutely essential feature for large yet flexible codebases

Mutagen tries to take the pleasure of writing and reading Python programs and extend it to "mathematically pure" programming, where the compiler can much more easily reason about the programs (which is very difficult for Python).

Current State
-------------

Mutagen is still in a very prototypical state. It is almost entirely implemented as a Python bootstrap program. The goal is to keep the bootstrap as small as possible, while allowing for the duplication of the bootstrap, and later compilation stages, in Mutagen, so as to allow for Pythonless execution. The bootstrap is a very simple interpreter, and actually keeps considerable state in the form of symbol tables and Python-backed data structures. This is just for simplicity in the initial implementation, and the language itself has no mutable state.

As of now, the Mutagen self-hosting compiler has a lexer, most of a parser, a basic assembler, and a very bare-bones register allocator. It is quite reasonable in terms of code size and readability, but its speed is still very much lacking due to the double interpreter layer (Mutagen on top of Python), and many non-optimal aspects of the current implementation (in the name of simplicity/laziness).
