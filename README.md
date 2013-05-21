Mutagen Programming Language
============================

The Mutagen programming language is a simple, readable purely functional language. Its syntax is very much influenced by Python, and is almost identical at the moment, though it will surely diverge in the future.

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
* Unbounded compilation times, to allow for more expensive search-based compiler algorithms (ideally ones that scale down on the low end as well, for fast minimal compilation times).
* Extensive caching in the compiler implementation, made possible by referential transparency, enabling the benefits of long compilation times on changing-but-still-similar codebases without the long latency.
* Better engineered stack. This remains to be seen, but the hope is that removing years of cruft from other languages will allow Mutagen to have a better rate of improvement in general.
* In the future, perhaps integrated microarchitectural simulations for hardware-feedback assisted tuning, or eventually fully integrated hardware/software codesign.

With readability, smart compilers, and superfast performance as fundamental language design choices, the overall goal of Mutagen is to increase programmer/computer efficiency exponentially, eventually leading to a blissful Utopian society free of want.

Current State
=============

Mutagen is still in a very prototypical state. It is almost entirely implemented as a Python bootstrap program, using PLY to support its lexing/parsing needs. The goal is to keep the bootstrap as small as possible, while allowing for the duplication of the bootstrap, and later compilation stages, in Mutagen, so as to allow for Pythonless execution. The bootstrap is a very simple interpreter, and actually keeps considerable state in the form of symbol tables. This is just for simplicity in the initial implementation, and the language itself has no mutable state.

As of now, only the lexer has an equivalent implementation in Mutagen. It is quite reasonable in terms of code size and readability when compared to the Python version, but its speed is still quite lacking due to the slow interpretive regex engine, as well as the double interpreter layer (Mutagen on top of Python).
