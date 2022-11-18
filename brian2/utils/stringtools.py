"""
A collection of tools for string formatting tasks.
"""
import re
import string

__all__ = [
    "indent",
    "deindent",
    "word_substitute",
    "replace",
    "get_identifiers",
    "strip_empty_lines",
    "stripped_deindented_lines",
    "strip_empty_leading_and_trailing_lines",
    "code_representation",
    "SpellChecker",
]


def indent(text, numtabs=1, spacespertab=4, tab=None):
    """
    Indents a given multiline string.

    By default, indentation is done using spaces rather than tab characters.
    To use tab characters, specify the tab character explictly, e.g.::

        indent(text, tab='\t')

    Note that in this case ``spacespertab`` is ignored.

    Examples
    --------
    >>> multiline = '''def f(x):
    ...     return x*x'''
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
    """
    if tab is None:
        tab = " " * spacespertab
    indent = tab * numtabs
    indentedstring = indent + text.replace("\n", f"\n{indent}")
    return indentedstring


def deindent(text, numtabs=None, spacespertab=4, docstring=False):
    """
    Returns a copy of the string with the common indentation removed.

    Note that all tab characters are replaced with ``spacespertab`` spaces.

    If the ``docstring`` flag is set, the first line is treated differently and
    is assumed to be already correctly tabulated.

    If the ``numtabs`` option is given, the amount of indentation to remove is
    given explicitly and not the common indentation.

    Examples
    --------
    Normal strings, e.g. function definitions:

    >>> multiline = '''    def f(x):
    ...          return x**2'''
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

    >>> docstring = '''First docstring line.
    ...     This line determines the indentation.'''
    >>> print(docstring)
    First docstring line.
        This line determines the indentation.
    >>> print(deindent(docstring, docstring=True))
    First docstring line.
    This line determines the indentation.
    """
    text = text.replace("\t", " " * spacespertab)
    lines = text.split("\n")
    # if it's a docstring, we search for the common tabulation starting from
    # line 1, otherwise we use all lines
    if docstring:
        start = 1
    else:
        start = 0
    if docstring and len(lines) < 2:  # nothing to do
        return text
    # Find the minimum indentation level
    if numtabs is not None:
        indentlevel = numtabs * spacespertab
    else:
        lineseq = [
            len(line) - len(line.lstrip())
            for line in lines[start:]
            if len(line.strip())
        ]
        if len(lineseq) == 0:
            indentlevel = 0
        else:
            indentlevel = min(lineseq)
    # remove the common indentation
    lines[start:] = [line[indentlevel:] for line in lines[start:]]
    return "\n".join(lines)


def word_substitute(expr, substitutions):
    """
    Applies a dict of word substitutions.

    The dict ``substitutions`` consists of pairs ``(word, rep)`` where each
    word ``word`` appearing in ``expr`` is replaced by ``rep``. Here a 'word'
    means anything matching the regexp ``\\bword\\b``.

    Examples
    --------

    >>> expr = 'a*_b+c5+8+f(A)'
    >>> print(word_substitute(expr, {'a':'banana', 'f':'func'}))
    banana*_b+c5+8+func(A)
    """
    for var, replace_var in substitutions.items():
        expr = re.sub(f"\\b{var}\\b", str(replace_var), expr)
    return expr


def replace(s, substitutions):
    """
    Applies a dictionary of substitutions. Simpler than `word_substitute`, it
    does not attempt to only replace words
    """
    for before, after in substitutions.items():
        s = s.replace(before, after)
    return s


KEYWORDS = {"and", "or", "not", "True", "False"}


def get_identifiers(expr, include_numbers=False):
    """
    Return all the identifiers in a given string ``expr``, that is everything
    that matches a programming language variable like expression, which is
    here implemented as the regexp ``\\b[A-Za-z_][A-Za-z0-9_]*\\b``.

    Parameters
    ----------
    expr : str
        The string to analyze
    include_numbers : bool, optional
        Whether to include number literals in the output. Defaults to ``False``.

    Returns
    -------
    identifiers : set
        A set of all the identifiers (and, optionally, numbers) in `expr`.

    Examples
    --------
    >>> expr = '3-a*_b+c5+8+f(A - .3e-10, tau_2)*17'
    >>> ids = get_identifiers(expr)
    >>> print(sorted(list(ids)))
    ['A', '_b', 'a', 'c5', 'f', 'tau_2']
    >>> ids = get_identifiers(expr, include_numbers=True)
    >>> print(sorted(list(ids)))
    ['.3e-10', '17', '3', '8', 'A', '_b', 'a', 'c5', 'f', 'tau_2']
    """
    identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr))
    if include_numbers:
        # only the number, not a + or -
        numbers = set(
            re.findall(
                r"(?<=[^A-Za-z_])[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|^[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?",
                expr,
            )
        )
    else:
        numbers = set()
    return (identifiers - KEYWORDS) | numbers


def strip_empty_lines(s):
    """
    Removes all empty lines from the multi-line string `s`.

    Examples
    --------

    >>> multiline = '''A string with
    ...
    ... an empty line.'''
    >>> print(strip_empty_lines(multiline))
    A string with
    an empty line.
    """
    return "\n".join(line for line in s.split("\n") if line.strip())


def strip_empty_leading_and_trailing_lines(s):
    """
    Removes all empty leading and trailing lines in the multi-line string `s`.
    """
    lines = s.split("\n")
    while lines and not lines[0].strip():
        del lines[0]
    while lines and not lines[-1].strip():
        del lines[-1]
    return "\n".join(lines)


def stripped_deindented_lines(code):
    """
    Returns a list of the lines in a multi-line string, deindented.
    """
    code = deindent(code)
    code = strip_empty_lines(code)
    lines = code.split("\n")
    return lines


def code_representation(code):
    """
    Returns a string representation for several different formats of code

    Formats covered include:
    - A single string
    - A list of statements/strings
    - A dict of strings
    - A dict of lists of statements/strings
    """
    if not isinstance(code, (str, list, tuple, dict)):
        code = str(code)
    if isinstance(code, str):
        return strip_empty_leading_and_trailing_lines(code)
    if not isinstance(code, dict):
        code = {None: code}
    else:
        code = code.copy()
    for k, v in code.items():
        if isinstance(v, (list, tuple)):
            v = "\n".join([str(line) for line in v])
            code[k] = v
    if len(code) == 1 and list(code.keys())[0] is None:
        return strip_empty_leading_and_trailing_lines(list(code.values())[0])
    output = []
    for k, v in code.items():
        msg = f"Key {k}:\n"
        msg += indent(str(v))
        output.append(msg)
    return strip_empty_leading_and_trailing_lines("\n".join(output))


# The below is adapted from Peter Norvig's spelling corrector
# http://norvig.com/spell.py (MIT licensed)
class SpellChecker(object):
    """
    A simple spell checker that will be used to suggest the correct name if the
    user made a typo (e.g. for state variable names).

    Parameters
    ----------
    words : iterable of str
        The known words
    alphabet : iterable of str, optional
        The allowed characters. Defaults to the characters allowed for
        identifiers, i.e. ascii characters, digits and the underscore.
    """

    def __init__(self, words, alphabet=f"{string.ascii_lowercase + string.digits}_"):
        self.words = words
        self.alphabet = alphabet

    def edits1(self, word):
        s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in s if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in s for c in self.alphabet if b]
        inserts = [a + c + b for a, b in s for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(
            e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.words
        )

    def known(self, words):
        return set(w for w in words if w in self.words)

    def suggest(self, word):
        return self.known(self.edits1(word)) or self.known_edits2(word) or set()
