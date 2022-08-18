import sympy as sp

class AlgebraicEquationSolver():

    def __init__(self):
        pass
    def solve(self,x,values_dict,eq):
        symbols_lst = sp.symbols(list(values_dict))
        symbols_lst
        x = sp.parse_expr(x)
        solve_lst = []
        solve_lst += [sp.parse_expr(eq)]

        for key, value in values_dict.items():
            solve_lst.append(sp.Eq(sp.parse_expr(key), value))
        try:
            return sp.solve(solve_lst)[0][x]
        except IndexError:
            print('This equation has no solutions.')