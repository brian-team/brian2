from brian2.utils.stringtools import indent, deindent

__all__ = ['codeprint']

def codeprint(code):
    if isinstance(code, str):
        print deindent(code)
    else:
        for k, v in code.items():
            print k+':'
            print indent(deindent(v))