"""
A collection of tools for string formatting tasks.
"""

import re

__all__ = ['indent',
           'deindent',
           'word_substitute',
           'get_identifiers',
           'strip_empty_lines',
           ]

def indent(text, numtabs=1, spacespertab=4, tab=None):
    """
    Indents a given multiline string.
    
    By default, indentation is done using spaces rather than tab characters.
    To use tab characters, specify the tab character explictly, e.g.::
    
        indent(text, tab='\t')
        
    Note that in this case ``spacespertab`` is ignored.
    """
    if tab is None:
        tab = ' '*spacespertab
    indent = tab*numtabs
    indentedstring = indent+text.replace('\n', '\n'+indent)
    return indentedstring

def deindent(text, numtabs=None, spacespertab=4, docstring=False):
    """
    Returns a copy of the string with the common indentation removed.
    
    Note that all tab characters are replaced with ``spacespertab`` spaces.
    
    For example for the following string (where # represents white space)::
    
        ####def f(x):
        ########return x**2
        
    It would return::
    
        def f(x):
        ####return x**2
        
    If the ``docstring`` flag is set, the first line is treated differently and
    is assumed to be already correctly tabulated.
    
    If the ``numtabs`` option is given, the amount of indentation to remove is
    given explicitly and not the common indentation.
    """
    text = text.replace('\t', ' '*spacespertab)
    lines = text.split('\n')
    # if it's a docstring, we search for the common tabulation starting from
    # line 1, otherwise we use all lines
    if docstring:
        start = 1
    else:
        start = 0
    if docstring and len(lines)<2: # nothing to do
        return text
    # Find the minimum indentation level
    if numtabs is not None:
        indentlevel = numtabs*spacespertab
    else:
        lineseq = [len(line)-len(line.lstrip()) for line in lines[start:] if len(line.strip())]
        if len(lineseq)==0:
            indentlevel = 0
        else:
            indentlevel = min(lineseq)
    # remove the common indentation
    lines[start:] = [line[indentlevel:] for line in lines[start:]]
    return '\n'.join(lines)

def word_substitute(expr, substitutions):
    '''
    Applies a dict of word substitutions.
    
    The dict ``substitutions`` consists of pairs ``(word, rep)`` where each
    word ``word`` appearing in ``expr`` is replaced by ``rep``. Here a 'word'
    means anything matching the regexp ``\\bword\\b``.
    
    For example::
    
        >>> expr = 'a*_b+c5+8+f(A)'
        >>> print word_substitute(expr, {'a':'banana', 'f':'func'})
        banana*_b+c5+8+func(A)   
    '''
    for var, replace_var in substitutions.iteritems():
        expr = re.sub(r'\b' + var + r'\b', str(replace_var), expr)
    return expr

def get_identifiers(expr):
    '''
    Return all the identifiers in a given string ``expr``, that is everything
    that matches a programming language variable like expression, which is
    here implemented as the regexp ``\\b[A-Za-z_][A-Za-z0-9_]*\\b``.
    
    For example::
    
        >>> expr = 'a*_b+c5+8+f(A)'
        >>> print get_identifiers(expr)
        ['a', '_b', 'c5', 'f', 'A']
    
    '''
    return re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', expr)

def strip_empty_lines(s):
    '''
    Removes all empty lines from the multi-line string ``s``.
    '''
    return '\n'.join(line for line in s.split('\n') if line.strip())

if __name__=='__main__':
    text = '''
    def f(x):
        return x*x
    '''
    print deindent(text)
    print indent(deindent(text).strip(), 1)
    print indent(deindent(text).strip(), 2, tab='#')
    print deindent(indent(text, 4), 2)
    text = '''A docstring
    With some text.
    '''
    print deindent(text, docstring=True)
    print deindent(text)
    