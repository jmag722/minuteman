import sympy as sp

class AlgebraicEquationSolver():
    """
    `AlgebraicEquationSolver` uses sympy to solve equations.
    """
    def __init__(self):
        pass
    def solve(self,x:str,values:dict,eq:str):
        """
        `solve` uses sympy.solve to solve equations.

        Parameters:
        x (str): value to be converted to symbol and solved.
        values (dict): Values of known symbols in equation.
        eq (str): Equation to be solved.
        Returns:
        x (float): Value of desired variable.
        """
        symbols_lst = sp.symbols(list(values))
        symbols_lst
        x = sp.parse_expr(x)
        solve_lst = []
        solve_lst += [sp.parse_expr(eq)]

        for key, value in values.items():
            solve_lst.append(sp.Eq(sp.parse_expr(key), value))
        try:
            return sp.solve(solve_lst)[0][x]
        except IndexError:
            print('This equation has no solutions.')