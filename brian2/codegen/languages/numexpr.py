from base import Language
from python import PythonLanguage

__all__ = ['NumexprPythonLanguage']

class NumexprPythonLanguage(PythonLanguage):
    def translate_expression(self, expr):
        # TODO: optimisation, heuristic for when it is better or worse to
        # use numexpr
        return '_numexpr.evaluate("'+expr.strip()+'")'
