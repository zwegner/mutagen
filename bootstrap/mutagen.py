#!/usr/bin/env python3
import inspect
import sys

import parse
import syntax

# Evaluate a single statement in a given context. Mainly a wrapper for error handling code.
def eval_statement(stmt, ctx):
    try:
        result = stmt.eval(ctx)
        return (result, False)
    except syntax.ProgramError as e:
        if e.stack_trace:
            for line in e.stack_trace:
                print(line, file=sys.stderr)
        print('%s(%i): %s' % (e.info.filename, e.info.lineno, e.msg), file=sys.stderr)
        return (None, True)
    except Exception as e:
        # All other exceptions shouldn't happen in theory, but they still do, so reconstruct
        # as much of the stack from Mutagen-space as we can
        traceback = inspect.trace()
        print('---------- INTERNAL COMPILER ERROR! ----------')
        print('Mutagen traceback:')
        last_index = 0
        for i, (frame, *_) in enumerate(traceback):
            # XXX this is pretty hacky--detect function calls by inspecting locals on the stack
            self = frame.f_locals.get('self', None)
            child_ctx = frame.f_locals.get('ctx', None)
            if isinstance(self, Call) and child_ctx:
                print('File "%s", line %s, in %s' % (self.info.filename, self.info.lineno, child_ctx.name))
                last_index = i
        # Below the part of the stack corresponding to the last Mutagen function call, print out
        # all the Python stack frames
        print('----------')
        print('Python traceback (below Mutagen traceback):')
        for frame, filename, lineno, function, code_context, index in traceback[last_index:]:
                print('  File "%s", line %s, in %s' % (filename, lineno, function))
                print('    ', code_context[index].strip())
        print('%s: %s' % (type(e).__name__, e))
        return (None, True)

def interpret(path):
    ctx = syntax.Context('__main__', None, None)
    ctx.fill_in_builtins()
    try:
        stmts = parse.parse(path, eval_ctx=ctx)
    except parse.libparse.ParseError as e:
        e.print()
        sys.exit(1)
    block = syntax.Block(stmts, info=syntax.BUILTIN_INFO)
    block = parse.preprocess_program(ctx, block)
    (result, error) = eval_statement(block, ctx)
    if error:
        sys.exit(1)

def repl():
    print('Mutagen v0.0.0, (c)2013-2019 Zach Wegner')

    eval_ctx = syntax.Context('__main__', None, None)
    eval_ctx.fill_in_builtins()
    parse_ctx = parse.ParseContext()

    # Add all the builtins from the __builtins__.mg file
    imp = parse.get_builtins_import()
    parse.handle_import(imp, parse_ctx)
    imp = parse.preprocess_program(eval_ctx, imp)
    imp.eval(eval_ctx)

    while True:
        # Read lines until the parser hits an error or has parsed a full statement
        line = ''
        try:
            while True:
                prompt = '>>> ' if not line else '... '
                line = line + input(prompt) + '\n'
                tokens = parse.tokenizer.input(line, filename='<stdin>', interactive=True)

                parse_ctx.all_imports = []
                stmt = parse.parser.parse(tokens, start='single_input', user_context=parse_ctx, lazy=True)
                if stmt:
                    break

            # Recursively parse imports. Do this here so we can catch ParseErrors
            for imp in parse_ctx.all_imports:
                parse.handle_import(imp, parse_ctx, eval_ctx=eval_ctx)
        except EOFError:
            print()
            break
        except parse.libparse.ParseError as e:
            e.print()
            continue

        # Preprocess the statement, run it, and print the result, if any
        stmt = parse.preprocess_program(eval_ctx, stmt)
        (result, error) = eval_statement(stmt, eval_ctx)
        if not error and result is not syntax.NONE:
            print(result.repr(eval_ctx))

def main(args):
    if len(args) == 1:
        repl()
        return
    assert len(args) == 2
    interpret(args[1])

if __name__ == '__main__':
    main(sys.argv)
