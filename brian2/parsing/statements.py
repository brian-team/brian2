from pyparsing import (
    CharsNotIn,
    Combine,
    Optional,
    ParseException,
    Regex,
    Suppress,
    Word,
    alphas,
    nums,
)

from brian2.utils.caching import cached

VARIABLE = Word(f"{alphas}_", f"{alphas + nums}_").setResultsName("variable")

OP = Regex(r"(\+|\-|\*|/|//|%|\*\*|>>|<<|&|\^|\|)?=").setResultsName("operation")
EXPR = Combine(
    CharsNotIn("=", min=1, max=1) + Optional(CharsNotIn("#"))
).setResultsName("expression")
COMMENT = Optional(CharsNotIn("#")).setResultsName("comment")
STATEMENT = VARIABLE + OP + EXPR + Optional(Suppress("#") + COMMENT)


@cached
def parse_statement(code):
    """
    parse_statement(code)

    Parses a single line of code into "var op expr".

    Parameters
    ----------
    code : str
        A string containing a single statement of the form
        ``var op expr # comment``, where the ``# comment`` part is optional.

    Returns
    -------
    var, op, expr, comment : str, str, str, str
        The four parts of the statement.

    Examples
    --------
    >>> parse_statement('v = -65*mV  # reset the membrane potential')
    ('v', '=', '-65*mV', 'reset the membrane potential')
    >>> parse_statement('v += dt*(-v/tau)')
    ('v', '+=', 'dt*(-v/tau)', '')
    """
    try:
        parsed = STATEMENT.parseString(code, parseAll=True)
    except ParseException as p_exc:
        raise ValueError(
            "Parsing the statement failed: \n"
            + str(p_exc.line)
            + "\n"
            + " " * (p_exc.column - 1)
            + "^\n"
            + str(p_exc)
        )
    parsed_statement = (
        parsed["variable"].strip(),
        parsed["operation"],
        parsed["expression"].strip(),
        parsed.get("comment", "").strip(),
    )

    return parsed_statement
