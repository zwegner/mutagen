#!/usr/bin/env python3
import mg_builtins
import parse
import syntax

def repl():
    eval_ctx = syntax.Context('__main__', None, None)
    parse_ctx = parse.ParseContext()

    # Add all the builtins, first from the __builtins__.mg file
    imp = parse.get_builtins_import()
    parse.handle_import(imp, parse_ctx)
    syntax.preprocess_program(eval_ctx, [imp])
    imp.eval(eval_ctx)

    # ...and then those defined in mg_builtins.py
    for k, v in mg_builtins.builtins.items():
        stmt = syntax.Assignment(syntax.Target([k], info=syntax.BUILTIN_INFO), v)
        stmt.eval(eval_ctx)

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
        syntax.preprocess_program(eval_ctx, [stmt])
        (result, error) = parse.eval_statement(stmt, eval_ctx)
        if not error and not isinstance(result, syntax.None_):
            print(result.str(eval_ctx))

if __name__ == '__main__':
    print('Mutagen v0.0.0, (c)2013-2018 Zach Wegner')
    repl()