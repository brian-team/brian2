import jinja2

code = '''
{{ uses_variables('a, b') }}
{{ uses_variables('a', 'b') }}
x
    y
    {{test|autoindent}}
    z
w
'''

lines = '''
a
   b
c
'''

def uses_variables(*args):
    s = ', '.join(args)
    return 'USES_VARIABLES{'+s+'}'

env = jinja2.Environment()
env.filters.update({'autoindent': lambda s: '%%START_AUTOINDENT%%'+s+'%%END_AUTOINDENT%%'})
env.globals.update({'uses_variables': uses_variables})
tmp = env.from_string(code)

def postfilter(s):
    lines = s.split('\n')
    outlines = []
    addspaces = 0
    for line in lines:
        if '%%START_AUTOINDENT%%' in line:
            if addspaces>0:
                raise SyntaxError("Cannot nest autoindents")
            addspaces = line.find('%%START_AUTOINDENT%%')
            line = line.replace('%%START_AUTOINDENT%%', '')
        if '%%END_AUTOINDENT%%' in line:
            line = line.replace('%%END_AUTOINDENT%%', '')
            addspaces = 0
        outlines.append(' '*addspaces+line)
    return '\n'.join(outlines)

print postfilter(tmp.render(test=lines))
