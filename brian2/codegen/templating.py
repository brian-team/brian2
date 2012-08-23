'''
This code is designed to handle the insertion of syntactically valid code
into another segment of code, preserving indentation (important for Python).
'''

from brian2.utils.stringtools import indent, deindent, strip_empty_lines

__all__ = ['apply_code_template']

def apply_code_template(code, template, placeholder='%CODE%'):
    '''
    Inserts the string ``code`` into ``template`` at ``placeholder``.
    
    The ``code`` is deindented, and inserted into ``template`` with the
    indentation level at the place where ``placeholder`` appears. The
    placeholder should appear on its own line.
    
    All tab characters are replaced by four spaces.
    '''
    code = deindent(code)
    #code = '\n'.join(line for line in code.split('\n') if line.strip())
    code = strip_empty_lines(code)
    template = template.replace('\t', ' '*4)
    lines = template.split('\n')
    newlines = []
    for line in lines:
        if placeholder in line:
            indentlevel = len(line)-len(line.lstrip())
            newlines.append(indent(code, indentlevel, tab=' '))
        else:
            newlines.append(line)
    return '\n'.join(newlines)

if __name__=='__main__':
    code = '''
    if cond:
        do_something()
    '''
    template = '''
    for arg in args:
        cond = f(arg)
        %CODE%
        do_something_else(arg)
    '''
    
    print deindent(strip_empty_lines(apply_code_template(code, template)))
