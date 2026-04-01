from pyparsing import (
    CharsNotIn,
    Combine,
    Optional,
    ParseException,
    ParseSyntaxException,
    Regex,
    Suppress,
    Word,
    alphas,
    nums,
    restOfLine,
)

from brian2.utils.caching import cached

VARIABLE = Word(f"{alphas}_", f"{alphas + nums}_").set_results_name("variable")

OP = (
    Regex(r"(\+|\-|\*|/|//|%|\*\*|>>|<<|&|\^|\|)?=")
    .set_results_name("operation")
    .set_name("assignment operator")
)
EXPR = Combine(
    CharsNotIn("=#", min=1, max=1) + Optional(CharsNotIn("#"))
).set_results_name("expression")
COMMENT = restOfLine.set_results_name("comment")
STATEMENT = VARIABLE - OP - EXPR + Optional(Suppress("#") + COMMENT)


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
        parsed = STATEMENT.parse_string(code, parseAll=True)
    except (ParseException, ParseSyntaxException) as p_exc:
        raise ValueError(
            f"Parsing the statement failed: {p_exc.msg}\n"
            + str(p_exc.line)
            + "\n"
            + " " * (p_exc.column - 1)
            + "^\n"
            + f"(line {p_exc.lineno}, col {p_exc.column})"
        ) from p_exc

    var = parsed["variable"].strip()
    op = parsed["operation"]
    expr = parsed["expression"].strip()
    comment = parsed.get("comment", "").strip()

    if expr.count("(") != expr.count(")"):
        raise ValueError(
            f"Unbalanced parentheses in expression: '{expr}'\n"
            f"{code}\n"
            f"{' ' * code.find(expr)}{'^' * len(expr)}"
        )

    return var, op, expr, comment
