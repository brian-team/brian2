"""
A collection of tools for string formatting tasks.
"""

import re

__all__ = ['indent',
           'deindent',
           'word_substitute',
           'get_identifiers',
           'strip_empty_lines',
           'stripped_deindented_lines',
           ]

def indent(text, numtabs=1, spacespertab=4, tab=None):
    '''
    Indents a given multiline string.
    
    By default, indentation is done using spaces rather than tab characters.
    To use tab characters, specify the tab character explictly, e.g.::
    
        indent(text, tab='\t')
        
    Note that in this case ``spacespertab`` is ignored.
    
    Examples
    --------
    >>> multiline = """def f(x):
    ...     return x*x"""
    >>> print(multiline)
    def f(x):
        return x*x
    >>> print(indent(multiline))
        def f(x):
            return x*x
    >>> print(indent(multiline, numtabs=2))
            def f(x):
                return x*x
    >>> print(indent(multiline, spacespertab=2))
      def f(x):
          return x*x
    >>> print(indent(multiline, tab='####'))
    ####def f(x):
    ####    return x*x
    '''
    if tab is None:
        tab = ' '*spacespertab
    indent = tab*numtabs
    indentedstring = indent+text.replace('\n', '\n'+indent)
    return indentedstring

def deindent(text, numtabs=None, spacespertab=4, docstring=False):
    '''
    Returns a copy of the string with the common indentation removed.
    
    Note that all tab characters are replaced with ``spacespertab`` spaces.
        
    If the ``docstring`` flag is set, the first line is treated differently and
    is assumed to be already correctly tabulated.
    
    If the ``numtabs`` option is given, the amount of indentation to remove is
    given explicitly and not the common indentation.
    
    Examples
    --------
    Normal strings, e.g. function definitions:
    
    >>> multiline = """    def f(x):
    ...          return x**2"""
    >>> print(multiline)
        def f(x):
             return x**2
    >>> print(deindent(multiline))
    def f(x):
         return x**2
    >>> print(deindent(multiline, docstring=True))
        def f(x):
    return x**2
    >>> print(deindent(multiline, numtabs=1, spacespertab=2))
      def f(x):
           return x**2
    
    Docstrings:
    
    >>> docstring = """First docstring line.
    ...     This line determines the indentation."""
    >>> print(docstring)
    First docstring line.
        This line determines the indentation.
    >>> print(deindent(docstring, docstring=True))
    First docstring line.
    This line determines the indentation.
    '''
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
    
    Examples
    --------
    
    >>> expr = 'a*_b+c5+8+f(A)'
    >>> print(word_substitute(expr, {'a':'banana', 'f':'func'}))
    banana*_b+c5+8+func(A)
    '''
    for var, replace_var in substitutions.iteritems():
        expr = re.sub(r'\b' + var + r'\b', str(replace_var), expr)
    return expr


KEYWORDS = set(['and', 'or', 'not', 'True', 'False'])

def get_identifiers(expr):
    '''
    Return all the identifiers in a given string ``expr``, that is everything
    that matches a programming language variable like expression, which is
    here implemented as the regexp ``\\b[A-Za-z_][A-Za-z0-9_]*\\b``.
    
    Examples
    --------
    
    >>> expr = 'a*_b+c5+8+f(A)'
    >>> ids = get_identifiers(expr)
    >>> print(sorted(list(ids)))
    ['A', '_b', 'a', 'c5', 'f']
    '''
    identifiers = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', expr))
    return identifiers - KEYWORDS

def strip_empty_lines(s):
    '''
    Removes all empty lines from the multi-line string `s`.
    
    Examples
    --------
    
    >>> multiline = """A string with
    ... 
    ... an empty line."""
    >>> print(strip_empty_lines(multiline))
    A string with
    an empty line.
    '''
    return '\n'.join(line for line in s.split('\n') if line.strip())

def stripped_deindented_lines(code):
    '''
    Returns a list of the lines in a multi-line string, deindented.
    '''
    code = deindent(code)
    code = strip_empty_lines(code)
    lines = code.split('\n')
    return lines
