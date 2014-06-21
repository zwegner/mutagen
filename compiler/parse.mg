import lexer
import liblex
import libparse

from syntax import *

# XXX
NULL_INFO = liblex.Info('???', 0)

def reduce_binop(p):
    r = p[0]
    for item in p[1]:
        r = BinaryOp(item[0], r, item[1])
    return r

def reduce_list(p):
    return [p[0]] + [item[1] for item in p[1]]

# Imports
def parse_import(p, module, names, path):
    imp = Import([], module, names, path, False, info=p.get_info(0))
    # XXX
    #all_imports.append(imp)
    return Scope(imp)

rules = [
    # Atoms
    ['identifier', ['IDENTIFIER', lambda(p): Identifier(p[0], info=p.get_info(0))]],
    ['none', ['NONE', lambda(p): None_(info=p.get_info(0))]],
    ['boolean', ['TRUE|FALSE', lambda(p):
        Boolean({'True': True, 'False': False}[p[0]], info=p.get_info(0))]],
    ['integer', ['INTEGER', lambda(p): Integer(p[0], info=p.get_info(0))]],
    ['string', ['STRING', lambda(p): String(p[0], info=p.get_info(0))]],
    ['parenthesized', ['LPAREN test RPAREN', lambda(p): p[1]]],
    ['atom', 'identifier|none|boolean|integer|string|parenthesized|list_comp|' +
        'dict_comp|set_comp'],

    ['list_comp', ['LBRACKET [test (comp_iter+|(COMMA test)* [COMMA])] RBRACKET',
        def(p) {
            if p[1] {
                items = p[1];
                l = [items[0]];
                if items[1] {
                    if isinstance(items[1][0], CompIter) {
                        return Scope(ListComprehension(items[0], items[1]));
                    }
                    l = l + [item[1] for item in items[1][0]];
                }
                return List(l, info=p.get_info(0));
            }
            return List([], info=p.get_info(0));
        }]],

    ['dict_comp', ['LBRACE [test COLON test (comp_iter+|' +
        '(COMMA test COLON test)* [COMMA])] RBRACE',
        def(p) {
            if p[1] {
                items = p[1];
                [key, value] = [items[0], items[2]];
                d = {key: value};
                if items[3] {
                    items = items[3];
                    if isinstance(items[0], CompIter) {
                        return Scope(DictComprehension(key, value, items));
                    }
                    d = d + {item[1]: item[3] for item in items[0]};
                }
                return Dict(d, info=p.get_info(0));
            }
            return Dict({}, info=p.get_info(0));
        }]],

    ['set_comp', ['LBRACE test (COMMA test)* [COMMA] RBRACE',
        def(p) {
            set_items = [p[1]] + [p[1] for item in p[2]];
            return Call(Identifier('set', info=p.get_info(0)),
                [List(set_items, info=p.get_info(0))]);
        }]],

    ['arg', ['test [EQUALS test]',
        def(p) {
            if p[1] {
                assert isinstance(p[0], Identifier);
                return KeywordArg(p[0].name, p[1][1]);
            }
            return p[0];
        }]],

    ['subscript', ['[test] [COLON [test] [COLON [test]]]]',
        def(p) {
            [start, stop, step] = [None_(info=NULL_INFO)] * 3;
            if p[0] {
                if p[1] == None {
                    return lambda(expr): GetItem(expr, p[0]);
                }
                start = p[0];
            }
            assert p[1];
            if p[1][1] != None {
                stop = p[1][1];
            }
            if p[1][2] != None and p[1][2][1] != None {
                step = p[1][2][1];
            }
            return lambda(expr): Call(Identifier('slice', info=NULL_INFO),
                [expr, start, stop, step]);
        }]],

    # Function calls, subscripts, attribute accesses
    ['vararg', ['STAR test', lambda(p): VarArg(p[1])]],
    ['kwvararg', ['STAR_STAR test', lambda(p): KeywordVarArg(p[1])]],
    ['args', ['(arg|vararg|kwvararg) (COMMA (arg|vararg|kwvararg))*', reduce_list]],
    # Since the trailer rules don't have access to the left-hand side, return lambdas
    ['call', ['LPAREN [args] RPAREN', lambda(p): lambda(expr): Call(expr, p[1] or [])]],
    ['getitem', ['LBRACKET subscript RBRACKET', lambda(p): p[1]]],
    ['getattr', ['PERIOD IDENTIFIER', lambda(p): lambda(expr): GetAttr(expr, p[1])]],
    ['trailer', 'call|getitem|getattr'],

    ['power', ['atom trailer* [STAR_STAR factor]',
        def(p) {
            r = p[0];
            for trailer in p[1] {
                r = trailer(r);
            }
            if p[2] {
                r = BinaryOp('**', r, p[2][1]);
            }
            return r;
        }]],

    # Binary ops
    ['factor', 'power', ['(PLUS|MINUS|INVERSE) factor', lambda(p): UnaryOp(p[0], p[1])]],
    ['term', ['factor ((STAR|FLOORDIV|MODULO) factor)*', reduce_binop]],
    ['arith_expr', ['term ((PLUS|MINUS) term)*', reduce_binop]],
    ['shift_expr', ['arith_expr ((SHIFT_LEFT|SHIFT_RIGHT) arith_expr)*', reduce_binop]],
    ['and_expr', ['shift_expr (BIT_AND shift_expr)*', reduce_binop]],
    ['xor_expr', ['and_expr (BIT_XOR and_expr)*', reduce_binop]],
    ['or_expr', ['xor_expr (BIT_OR xor_expr)*', reduce_binop]],
    ['expr', 'or_expr'],

    # XXX chaining comparisons?
    ['comparison', ['or_expr ((EQUALS_EQUALS|NOT_EQUALS|' +
        'GREATER|GREATER_EQUALS|LESS|LESS_EQUALS|IN|NOT IN) or_expr)*',
        def(p) {
            r = p[0];
            for item in p[1] {
                if item[0] == 'in' {
                    r = BinaryOp('in', item[1], r);
                }
                elif item[0] == ['not', 'in'] {
                    r = UnaryOp('not', BinaryOp('in', item[1], r));
                }
                else {
                    r = BinaryOp(item[0], r, item[1]);
                }
            }
            return r;
        }]],

    ['not_test', 'comparison', ['NOT not_test', lambda(p): UnaryOp('not', p[1])]],
    ['and_test', ['not_test (AND not_test)*', reduce_binop]],
    ['or_test', ['and_test (OR and_test)*', reduce_binop]],
    ['test', 'or_test|lambdef|def_expr'],

    ['for_assn_base', 'IDENTIFIER', ['LBRACKET for_assn_list RBRACKET',
        lambda(p): p[1]]],
    ['for_assn_list', ['for_assn_base (COMMA for_assn_base)*', reduce_list]],
    ['for_assn', ['for_assn_base', lambda(p): Target([p[0]], info=NULL_INFO)]],
    ['comp_iter', ['FOR for_assn IN test', lambda(p): CompIter(p[1], p[3])]],

    # Statements
    ['pass', ['PASS', lambda(p): None]],
    ['small_stmt', '(expr_stmt|pass|break|continue|return|yield|import|assert)'],
    ['simple_stmt', ['small_stmt (NEWLINE|SEMICOLON)', lambda(p): p[0]]],
    ['stmt', ['(simple_stmt|if_stmt|for_stmt|while_stmt|def_stmt|class_stmt) ' +
        '(NEWLINE|SEMICOLON)*', lambda(p): p[0]]],
    ['stmt_list', ['stmt*', lambda(p): listlambda filter((x: x != None, p[0]))]],

    ['break', ['BREAK', lambda(p): Break(info=p.get_info(0))]],
    ['continue', ['CONTINUE', lambda(p): Continue(info=p.get_info(0))]],
    ['return', ['RETURN test', lambda(p): Return(p[1])]],
    ['yield', ['YIELD test', lambda(p): Yield(p[1])]],
    ['assert', ['ASSERT test', lambda(p): Assert(p[1])]],

    ['expr_stmt', ['test (EQUALS test)*',
        def(p) {
            @fixed_point;
            def deconstruct_lhs(deconstruct_lhs, lhs) {
                if isinstance(lhs, Identifier) {
                    return lhs.name;
                }
                elif isinstance(lhs, List) {
                    return [deconstruct_lhs(i) for i in lhs];
                }
                lhs.error('invalid lhs for assignment');
            }
            if p[1] {
                all_items = reduce_list(p);
                [targets, expr] = [all_items[:-1], all_items[-1]];
                return Assignment(Target([deconstruct_lhs(t) for t in targets],
                    info=NULL_INFO), expr);
            }
            return p[0];
        }]],

    # Blocks
    ['delims', ['NEWLINE+', lambda(p): None]],
    ['small_stmt_list', ['small_stmt (SEMICOLON small_stmt)*',
        lambda(p): listlambda filter((x: x != None, reduce_list(p)))]],
    ['block',
        ['COLON delims INDENT stmt_list DEDENT', lambda(p): Block(p[3], info=p.get_info(0))],
        ['COLON small_stmt_list NEWLINE', lambda(p): Block(p[1], info=p.get_info(0))],
        ['LBRACE stmt_list RBRACE', lambda(p): Block(p[1], info=p.get_info(0))]],
    ['for_stmt', ['FOR for_assn IN test block', lambda(p): For(p[1], p[3], p[4])]],
    ['while_stmt', ['WHILE test block', lambda(p): While(p[1], p[2])]],

    ['if_stmt', ['IF test block (ELIF test block)* [ELSE block]',
        def(p) {
            else_block = p[4][1] if p[4] else Block([], info=p.get_info(0));
            for elif_stmt in reversed(p[3]) {
                else_block = IfElse(elif_stmt[1], elif_stmt[2], else_block);
            }
            return IfElse(p[1], p[2], else_block);
        }]],

    # Params
    ['param', ['IDENTIFIER [COLON test] [EQUALS test]',
        lambda(p): [p[0], p[1][1] if p[1] else None, p[2][1] if p[2] else None]]],

    ['param', ['STAR IDENTIFIER', lambda(p): VarParams(p[1], info=p.get_info(0))],
        ['STAR_STAR IDENTIFIER', lambda(p): KeywordVarParams(p[1], info=p.get_info(0))]],
    ['param_list', ['param (COMMA param)*', reduce_list]],

    ['params', ['[LPAREN [param_list] RPAREN]',
        def(p) {
            [params, types, var_params, kwparams, kw_var_params] = [[], [], None, [], None];
            if p[0] and p[0][1] {
                for item in p[0][1] {
                    if isinstance(item, VarParams) {
                        assert not var_params;
                        assert not kwparams;
                        assert not kw_var_params;
                        var_params = item.name;
                    } elif isinstance(item, KeywordVarParams) {
                        kw_var_params = item.name;
                    } elif item[2] {
                        assert not kw_var_params;
                        # XXX no typed keyword arguments yet
                        assert not item[1];
                        kwparams = kwparams + [KeywordArg(item[0], item[2])];
                    } else {
                        assert not var_params;
                        assert not kwparams;
                        assert not kw_var_params;
                        params = params + [item[0]];
                        types = types + [item[1] or None_(info=NULL_INFO)];
                    }
                }
            }
            return Params(params, types, var_params, kwparams, kw_var_params,
                info=NULL_INFO);
        }]],

    # Function/class defs
    ['decorator', ['AT test delims', lambda(p): p[1]]],
    ['return_type', ['[RARROW test]', lambda(p): p[0][1] if p[0] else None]],
    ['lambdef', ['LAMBDA params COLON test',
        lambda(p): Scope(Function('lambda', p[1], None,
            Block([Return(p[3])], info=p.get_info(0)), info=p.get_info(0)))]],
    ['def_expr', ['DEF params return_type block',
        lambda(p): Scope(Function('lambda', p[1], p[2], p[3], info=p.get_info(0)))]],
    ['class_stmt', ['CLASS IDENTIFIER params block',
        lambda(p): Assignment(Target([p[1]], info=p.get_info(1)),
            Scope(Class(p[1], p[2], p[3], info=p.get_info(0))))]],

    ['def_stmt', ['decorator* DEF IDENTIFIER params return_type block',
        def(p) {
            fn = Scope(Function(p[2], p[3], p[4], p[5], info=p.get_info(1)));
            for dec in p[0] {
                fn = Call(dec, [fn]);
            }
            return Assignment(Target([p[2]], info=p.get_info(2)), fn);
        }]],

    ['ident_list', ['IDENTIFIER (COMMA IDENTIFIER)*', reduce_list]],
    ['import', ['IMPORT IDENTIFIER [FROM STRING]',
            lambda(p): parse_import(p, p[1], None, p[2][1] if p[2] else None)],
        ['FROM IDENTIFIER IMPORT ident_list',
            lambda(p): parse_import(p, p[1], p[3], None)],
        ['FROM IDENTIFIER IMPORT STAR',
            lambda(p): parse_import(p, p[1], [], None)]],
]

parser = libparse.Parser(rules, 'stmt_list')
line = 'x(x.x() * x.x(x, x.x()))\n'
x = parser.parse(lexer.input(line))
print(x[0])
