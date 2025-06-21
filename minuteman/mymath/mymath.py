import sympy


def solve_algebraic_eqn(unknown: str, knowns: dict, eq: str):
    """
    Solve algebraic equation with sympy.solve

    Parameters
    ----------
    unknown : str
        value to be converted to symbol and solved.
    knowns : dict
        Key value pairs of known symbols in equation.
    eq : str
        Equation to be solved.

    Returns
    -------
    Any
        Value of desired variable.
    """
    symbols_lst = sympy.symbols(list(knowns))
    symbols_lst
    unknown = sympy.parse_expr(unknown)
    solve_lst = []
    solve_lst += [sympy.parse_expr(eq)]

    for key, value in knowns.items():
        solve_lst.append(sympy.Eq(sympy.parse_expr(key), value))
    try:
        return sympy.solve(solve_lst)[0][unknown]
    except IndexError:
        print('This equation has no solutions.')
