"""
Differential equations for Brian models.
"""

import keyword
import re
import string
from collections import namedtuple
from collections.abc import Hashable, Mapping

import sympy
from pyparsing import (
    CharsNotIn,
    Combine,
    Group,
    LineEnd,
    OneOrMore,
    Optional,
    ParseException,
    Suppress,
    Word,
    ZeroOrMore,
    restOfLine,
)

from brian2.core.namespace import DEFAULT_CONSTANTS, DEFAULT_FUNCTIONS, DEFAULT_UNITS
from brian2.parsing.sympytools import str_to_sympy, sympy_to_str
from brian2.units.allunits import second
from brian2.units.fundamentalunits import (
    DIMENSIONLESS,
    DimensionMismatchError,
    Quantity,
    Unit,
    get_dimensions,
    get_unit,
    get_unit_for_display,
)
from brian2.utils.caching import CacheKey, cached
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers
from brian2.utils.topsort import topsort

from .codestrings import Expression
from .unitcheck import check_dimensions

__all__ = ["Equations"]

logger = get_logger(__name__)

# Equation types (currently simple strings but always use the constants,
# this might get refactored into objects, for example)
PARAMETER = "parameter"
DIFFERENTIAL_EQUATION = "differential equation"
SUBEXPRESSION = "subexpression"

# variable types (FLOAT is the only one that is possible for variables that
# have dimensions). These types will be later translated into dtypes, either
# using the default values from the preferences, or explicitly given dtypes in
# the construction of the `NeuronGroup`, `Synapses`, etc. object
FLOAT = "float"
INTEGER = "integer"
BOOLEAN = "boolean"

# Definitions of equation structure for parsing with pyparsing
# TODO: Maybe move them somewhere else to not pollute the namespace here?
#       Only IDENTIFIER and EQUATIONS are ever used later
###############################################################################
# Basic Elements
###############################################################################

# identifiers like in C: can start with letter or underscore, then a
# combination of letters, numbers and underscores
# Note that the check_identifiers function later performs more checks, e.g.
# names starting with underscore should only be used internally
IDENTIFIER = Word(
    string.ascii_letters + "_", string.ascii_letters + string.digits + "_"
).setResultsName("identifier")

# very broad definition here, expression will be analysed by sympy anyway
# allows for multi-line expressions, where each line can have comments
EXPRESSION = Combine(
    OneOrMore(
        (CharsNotIn(":#\n") + Suppress(Optional(LineEnd()))).ignore("#" + restOfLine)
    ),
    joinString=" ",
).setResultsName("expression")

# a unit
# very broad definition here, again. Whether this corresponds to a valid unit
# string will be checked later
UNIT = Word(string.ascii_letters + string.digits + "*/.- ").setResultsName("unit")

# a single Flag (e.g. "const" or "event-driven")
FLAG = Word(string.ascii_letters, string.ascii_letters + "_- " + string.digits)

# Flags are comma-separated and enclosed in parantheses: "(flag1, flag2)"
FLAGS = (
    Suppress("(") + FLAG + ZeroOrMore(Suppress(",") + FLAG) + Suppress(")")
).setResultsName("flags")

###############################################################################
# Equations
###############################################################################
# Three types of equations
# Parameter:
# x : volt (flags)
PARAMETER_EQ = Group(
    IDENTIFIER + Suppress(":") + UNIT + Optional(FLAGS)
).setResultsName(PARAMETER)

# Static equation:
# x = 2 * y : volt (flags)
STATIC_EQ = Group(
    IDENTIFIER + Suppress("=") + EXPRESSION + Suppress(":") + UNIT + Optional(FLAGS)
).setResultsName(SUBEXPRESSION)

# Differential equation
# dx/dt = -x / tau : volt
DIFF_OP = Suppress("d") + IDENTIFIER + Suppress("/") + Suppress("dt")
DIFF_EQ = Group(
    DIFF_OP + Suppress("=") + EXPRESSION + Suppress(":") + UNIT + Optional(FLAGS)
).setResultsName(DIFFERENTIAL_EQUATION)

# ignore comments
EQUATION = (PARAMETER_EQ | STATIC_EQ | DIFF_EQ).ignore("#" + restOfLine)
EQUATIONS = ZeroOrMore(EQUATION)


class EquationError(Exception):
    """
    Exception type related to errors in an equation definition.
    """

    pass


def check_identifier_basic(identifier):
    """
    Check an identifier (usually resulting from an equation string provided by
    the user) for conformity with the rules. The rules are:

        1. Only ASCII characters
        2. Starts with a character, then mix of alphanumerical characters and
           underscore
        3. Is not a reserved keyword of Python

    Parameters
    ----------
    identifier : str
        The identifier that should be checked

    Raises
    ------
    SyntaxError
        If the identifier does not conform to the above rules.
    """

    # Check whether the identifier is parsed correctly -- this is always the
    # case, if the identifier results from the parsing of an equation but there
    # might be situations where the identifier is specified directly
    parse_result = list(IDENTIFIER.scanString(identifier))

    # parse_result[0][0][0] refers to the matched string -- this should be the
    # full identifier, if not it is an illegal identifier like "3foo" which only
    # matched on "foo"
    if len(parse_result) != 1 or parse_result[0][0][0] != identifier:
        raise SyntaxError(f"'{identifier}' is not a valid variable name.")

    if keyword.iskeyword(identifier):
        raise SyntaxError(
            f"'{identifier}' is a Python keyword and cannot be used as a variable."
        )

    if identifier.startswith("_"):
        raise SyntaxError(
            f"Variable '{identifier}' starts with an underscore, "
            "this is only allowed for variables used "
            "internally"
        )


def check_identifier_reserved(identifier):
    """
    Check that an identifier is not using a reserved special variable name. The
    special variables are: 't', 'dt', and 'xi', as well as everything starting
    with `xi_`.

    Parameters
    ----------
    identifier: str
        The identifier that should be checked

    Raises
    ------
    SyntaxError
        If the identifier is a special variable name.
    """
    if identifier in (
        "t",
        "dt",
        "t_in_timesteps",
        "xi",
        "i",
        "N",
    ) or identifier.startswith("xi_"):
        raise SyntaxError(
            f"'{identifier}' has a special meaning in equations and "
            "cannot be used as a variable name."
        )


def check_identifier_units(identifier):
    """
    Make sure that identifier names do not clash with unit names.
    """
    if identifier in DEFAULT_UNITS:
        raise SyntaxError(
            f"'{identifier}' is the name of a unit, cannot be used as a variable name."
        )


def check_identifier_functions(identifier):
    """
    Make sure that identifier names do not clash with function names.
    """
    if identifier in DEFAULT_FUNCTIONS:
        raise SyntaxError(
            f"'{identifier}' is the name of a function, cannot be used as "
            "a variable name."
        )


def check_identifier_constants(identifier):
    """
    Make sure that identifier names do not clash with function names.
    """
    if identifier in DEFAULT_CONSTANTS:
        raise SyntaxError(
            f"'{identifier}' is the name of a constant, cannot be used as "
            "a variable name."
        )


_base_units_with_alternatives = None
_base_units = None


def dimensions_and_type_from_string(unit_string):
    """
    Returns the physical dimensions that results from evaluating a string like
    "siemens / metre ** 2", allowing for the special string "1" to signify
    dimensionless units, the string "boolean" for a boolean and "integer" for
    an integer variable.

    Parameters
    ----------
    unit_string : str
        The string that should evaluate to a unit

    Returns
    -------
    d, type : (`Dimension`, {FLOAT, INTEGER or BOOL})
        The resulting physical dimensions and the type of the variable.

    Raises
    ------
    ValueError
        If the string cannot be evaluated to a unit.
    """
    # Lazy import to avoid circular dependency
    from brian2.core.namespace import DEFAULT_UNITS

    global _base_units_with_alternatives
    global _base_units
    if _base_units_with_alternatives is None:
        base_units_for_dims = {}
        for unit_name, unit in reversed(DEFAULT_UNITS.items()):
            if float(unit) == 1.0 and repr(unit)[-1] not in ["2", "3"]:
                if unit.dim in base_units_for_dims:
                    if unit_name not in base_units_for_dims[unit.dim]:
                        base_units_for_dims[unit.dim].append(unit_name)
                else:
                    base_units_for_dims[unit.dim] = [repr(unit)]
                    if unit_name != repr(unit):
                        base_units_for_dims[unit.dim].append(unit_name)
        alternatives = sorted(
            [tuple(values) for values in base_units_for_dims.values()]
        )
        _base_units = {v: DEFAULT_UNITS[v] for values in alternatives for v in values}
        # Create a string that lists all allowed base units
        alternative_strings = []
        for units in alternatives:
            string = units[0]
            if len(units) > 1:
                other_units = ", ".join(units[1:])
                string += f" ({other_units})"
            alternative_strings.append(string)
        _base_units_with_alternatives = ", ".join(alternative_strings)

    unit_string = unit_string.strip()

    # Special case: dimensionless unit
    if unit_string == "1":
        return DIMENSIONLESS, FLOAT

    # Another special case: boolean variable
    if unit_string == "boolean":
        return DIMENSIONLESS, BOOLEAN
    if unit_string == "bool":
        raise TypeError("Use 'boolean' not 'bool' as the unit for a boolean variable.")

    # Yet another special case: integer variable
    if unit_string == "integer":
        return DIMENSIONLESS, INTEGER

    # Check first whether the expression only refers to base units
    identifiers = get_identifiers(unit_string)
    for identifier in identifiers:
        if identifier not in _base_units:
            if identifier in DEFAULT_UNITS:
                # A known unit, but not a base unit
                base_unit = get_unit(DEFAULT_UNITS[identifier].dim)
                if not repr(base_unit) in _base_units:
                    # Make sure that we don't suggest a unit that is not allowed
                    # (should not happen, normally)
                    base_unit = Unit(1, dim=base_unit.dim)
                raise ValueError(
                    "Unit specification refers to "
                    f"'{identifier}', but this is not a base "
                    f"unit. Use '{base_unit!r}' instead."
                )
            else:
                # Not a known unit
                raise ValueError(
                    "Unit specification refers to "
                    f"'{identifier}', but this is not a base "
                    "unit. The following base units are "
                    f"allowed: {_base_units_with_alternatives}."
                )
    try:
        evaluated_unit = eval(unit_string, _base_units)
    except Exception as ex:
        raise ValueError(
            f"Could not interpret '{unit_string}' as a unit specification: {ex}"
        )

    # Check whether the result is a unit
    if not isinstance(evaluated_unit, Unit):
        if isinstance(evaluated_unit, Quantity):
            raise ValueError(
                f"'{unit_string}' does not evaluate to a unit but to a "
                "quantity -- make sure to only use units, e.g. "
                "'siemens/metre**2' and not '1 * siemens/metre**2'"
            )
        else:
            raise ValueError(
                f"'{unit_string}' does not evaluate to a unit, the result "
                f"has type {type(evaluated_unit)} instead."
            )

    # No error has been raised, all good
    return evaluated_unit.dim, FLOAT


@cached
def parse_string_equations(eqns):
    """
    parse_string_equations(eqns)

    Parse a string defining equations.

    Parameters
    ----------
    eqns : str
        The (possibly multi-line) string defining the equations. See the
        documentation of the `Equations` class for details.

    Returns
    -------
    equations : dict
        A dictionary mapping variable names to
        `~brian2.equations.equations.Equations` objects
    """
    equations = {}

    try:
        parsed = EQUATIONS.parseString(eqns, parseAll=True)
    except ParseException as p_exc:
        raise EquationError(
            "Parsing failed: \n"
            + str(p_exc.line)
            + "\n"
            + " " * (p_exc.column - 1)
            + "^\n"
            + str(p_exc)
        ) from p_exc
    for eq in parsed:
        eq_type = eq.getName()
        eq_content = dict(eq.items())
        # Check for reserved keywords
        identifier = eq_content["identifier"]

        # Convert unit string to Unit object
        try:
            dims, var_type = dimensions_and_type_from_string(eq_content["unit"])
        except ValueError as ex:
            raise EquationError(
                "Error parsing the unit specification for "
                f"variable '{identifier}': {ex}"
            )

        expression = eq_content.get("expression", None)
        if expression is not None:
            # Replace multiple whitespaces (arising from joining multiline
            # strings) with single space
            p = re.compile(r"\s{2,}")
            expression = Expression(p.sub(" ", expression))
        flags = list(eq_content.get("flags", []))

        equation = SingleEquation(
            eq_type, identifier, dims, var_type=var_type, expr=expression, flags=flags
        )

        if identifier in equations:
            raise EquationError(f"Duplicate definition of variable '{identifier}'")

        equations[identifier] = equation

    return equations


class SingleEquation(Hashable, CacheKey):
    """
    Class for internal use, encapsulates a single equation or parameter.

    .. note::
        This class should never be used directly, it is only useful as part of
        the `Equations` class.

    Parameters
    ----------
    type : {PARAMETER, DIFFERENTIAL_EQUATION, SUBEXPRESSION}
        The type of the equation.
    varname : str
        The variable that is defined by this equation.
    dimensions : `Dimension`
        The physical dimensions of the variable
    var_type : {FLOAT, INTEGER, BOOLEAN}
        The type of the variable (floating point value or boolean).
    expr : `Expression`, optional
        The expression defining the variable (or ``None`` for parameters).
    flags: list of str, optional
        A list of flags that give additional information about this equation.
        What flags are possible depends on the type of the equation and the
        context.
    """

    _cache_irrelevant_attributes = {"update_order"}

    def __init__(
        self, type, varname, dimensions, var_type=FLOAT, expr=None, flags=None
    ):
        self.type = type
        self.varname = varname
        self.dim = get_dimensions(dimensions)
        self.var_type = var_type
        if dimensions is not DIMENSIONLESS:
            if var_type == BOOLEAN:
                raise TypeError("Boolean variables are necessarily dimensionless.")
            elif var_type == INTEGER:
                raise TypeError("Integer variables are necessarily dimensionless.")

        if type == DIFFERENTIAL_EQUATION:
            if var_type != FLOAT:
                raise TypeError(
                    "Differential equations can only define floating point variables"
                )
        self.expr = expr
        if flags is None:
            self.flags = []
        else:
            self.flags = list(flags)

        # will be set later in the sort_subexpressions method of Equations
        self.update_order = -1

    unit = property(lambda self: get_unit(self.dim), doc="The `Unit` of this equation.")

    identifiers = property(
        lambda self: self.expr.identifiers if self.expr is not None else set(),
        doc="All identifiers in the RHS of this equation.",
    )

    stochastic_variables = property(
        lambda self: {
            variable
            for variable in self.identifiers
            if variable == "xi" or variable.startswith("xi_")
        },
        doc="Stochastic variables in the RHS of this equation",
    )

    def __eq__(self, other):
        if not isinstance(other, SingleEquation):
            return NotImplemented
        return self._state_tuple == other._state_tuple

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._state_tuple)

    def _latex(self, *args):
        if self.type == DIFFERENTIAL_EQUATION:
            return (
                r"\frac{\mathrm{d}"
                + sympy.latex(sympy.Symbol(self.varname))
                + r"}{\mathrm{d}t} = "
                + sympy.latex(str_to_sympy(self.expr.code))
            )
        elif self.type == SUBEXPRESSION:
            return (
                sympy.latex(sympy.Symbol(self.varname))
                + " = "
                + sympy.latex(str_to_sympy(self.expr.code))
            )
        elif self.type == PARAMETER:
            return sympy.latex(sympy.Symbol(self.varname))

    def __str__(self):
        if self.type == DIFFERENTIAL_EQUATION:
            s = "d" + self.varname + "/dt"
        else:
            s = self.varname

        if self.expr is not None:
            s += " = " + str(self.expr)

        s += " : " + get_unit_for_display(self.dim)

        if len(self.flags):
            s += " (" + ", ".join(self.flags) + ")"

        return s

    def __repr__(self):
        s = "<" + self.type + " " + self.varname

        if self.expr is not None:
            s += ": " + self.expr.code

        s += " (Unit: " + get_unit_for_display(self.dim)

        if len(self.flags):
            s += ", flags: " + ", ".join(self.flags)

        s += ")>"
        return s

    def _repr_pretty_(self, p, cycle):
        """
        Pretty printing for ipython.
        """
        if cycle:
            # should never happen
            raise AssertionError("Cyclical call of SingleEquation._repr_pretty")

        if self.type == DIFFERENTIAL_EQUATION:
            p.text("d" + self.varname + "/dt")
        else:
            p.text(self.varname)

        if self.expr is not None:
            p.text(" = ")
            p.pretty(self.expr)

        p.text(" : ")
        p.pretty(get_unit(self.dim))

        if len(self.flags):
            p.text(" (" + ", ".join(self.flags) + ")")

    def _repr_latex_(self):
        return "$" + sympy.latex(self) + "$"


class Equations(Hashable, Mapping):
    """
    Container that stores equations from which models can be created.

    String equations can be of any of the following forms:

    1. ``dx/dt = f : unit (flags)`` (differential equation)
    2. ``x = f : unit (flags)`` (equation)
    3. ``x : unit (flags)`` (parameter)

    String equations can span several lines and contain Python-style comments
    starting with ``#``

    Parameters
    ----------
    eqs : `str` or list of `SingleEquation` objects
        A multiline string of equations (see above) -- for internal purposes
        also a list of `SingleEquation` objects can be given. This is done for
        example when adding new equations to implement the refractory
        mechanism. Note that in this case the variable names are not checked
        to allow for "internal names", starting with an underscore.
    kwds: keyword arguments
        Keyword arguments can be used to replace variables in the equation
        string. Arguments have to be of the form ``varname=replacement``, where
        `varname` has to correspond to a variable name in the given equation.
        The replacement can be either a string (replacing a name with a new
        name, e.g. ``tau='tau_e'``) or a value (replacing the variable name
        with the value, e.g. ``tau=tau_e`` or ``tau=10*ms``).
    """

    def __init__(self, eqns, **kwds):
        if isinstance(eqns, str):
            self._equations = parse_string_equations(eqns)
            # Do a basic check for the identifiers
            self.check_identifiers()
        else:
            self._equations = {}
            for eq in eqns:
                if not isinstance(eq, SingleEquation):
                    raise TypeError(
                        "The list should only contain "
                        f"SingleEquation objects, not {type(eq)}"
                    )
                if eq.varname in self._equations:
                    raise EquationError(
                        f"Duplicate definition of variable '{eq.varname}'"
                    )
                self._equations[eq.varname] = eq

        self._equations = self._substitute(kwds)

        # Check for special symbol xi (stochastic term)
        uses_xi = None
        for eq in self._equations.values():
            if eq.expr is not None and "xi" in eq.expr.identifiers:
                if not eq.type == DIFFERENTIAL_EQUATION:
                    raise EquationError(
                        f"The equation defining '{eq.varname}' "
                        "contains the symbol 'xi' but is not a "
                        "differential equation."
                    )
                elif uses_xi is not None:
                    raise EquationError(
                        f"The equation defining {eq.varname} contains "
                        "the symbol 'xi', but it is already used "
                        f"in the equation defining {uses_xi}. Rename "
                        "the variables to 'xi_...' to make "
                        "clear whether they are the same or "
                        "independent random variables. Using "
                        "the same name twice will lead to "
                        "identical noise realizations "
                        "whereas using different names will "
                        "lead to independent noise "
                        "realizations."
                    )
                else:
                    uses_xi = eq.varname

        # rearrange subexpressions
        self._sort_subexpressions()

        #: Cache for equations with the subexpressions substituted
        self._substituted_expressions = None

    def _substitute(self, replacements):
        if len(replacements) == 0:
            return self._equations

        new_equations = {}
        for eq in self.values():
            # Replace the name of a model variable (works only for strings)
            if eq.varname in replacements:
                new_varname = replacements[eq.varname]
                if not isinstance(new_varname, str):
                    raise ValueError(
                        f"Cannot replace model variable '{eq.varname}' with a value."
                    )
                if new_varname in self or new_varname in new_equations:
                    raise EquationError(
                        f"Cannot replace model variable '{eq.varname}' "
                        f"with '{new_varname}', duplicate definition "
                        f"of '{new_varname}'."
                    )
                # make sure that the replacement is a valid identifier
                Equations.check_identifier(new_varname)
            else:
                new_varname = eq.varname

            if eq.type in [SUBEXPRESSION, DIFFERENTIAL_EQUATION]:
                # Replace values in the RHS of the equation
                new_code = eq.expr.code
                for to_replace, replacement in replacements.items():
                    if to_replace in eq.identifiers:
                        if isinstance(replacement, str):
                            # replace the name with another name
                            new_code = re.sub(
                                "\\b" + to_replace + "\\b", replacement, new_code
                            )
                        else:
                            # replace the name with a value
                            new_code = re.sub(
                                "\\b" + to_replace + "\\b",
                                "(" + repr(replacement) + ")",
                                new_code,
                            )
                        try:
                            Expression(new_code)
                        except ValueError as ex:
                            raise ValueError(
                                'Replacing "%s" with "%r" failed: %s'
                                % (to_replace, replacement, ex)
                            )
                new_equations[new_varname] = SingleEquation(
                    eq.type,
                    new_varname,
                    dimensions=eq.dim,
                    var_type=eq.var_type,
                    expr=Expression(new_code),
                    flags=eq.flags,
                )
            else:
                new_equations[new_varname] = SingleEquation(
                    eq.type,
                    new_varname,
                    dimensions=eq.dim,
                    var_type=eq.var_type,
                    flags=eq.flags,
                )

        return new_equations

    def substitute(self, **kwds):
        return Equations(list(self._substitute(kwds).values()))

    def __iter__(self):
        return iter(self._equations)

    def __len__(self):
        return len(self._equations)

    def __getitem__(self, key):
        return self._equations[key]

    def __add__(self, other_eqns):
        if isinstance(other_eqns, str):
            other_eqns = parse_string_equations(other_eqns)
        elif not isinstance(other_eqns, Equations):
            return NotImplemented

        return Equations(list(self.values()) + list(other_eqns.values()))

    def __hash__(self):
        return hash(frozenset(self._equations.items()))

    #: A set of functions that are used to check identifiers (class attribute).
    #: Functions can be registered with the static method
    #: `Equations.register_identifier_check` and will be automatically
    #: used when checking identifiers
    identifier_checks = {
        check_identifier_basic,
        check_identifier_reserved,
        check_identifier_functions,
        check_identifier_constants,
        check_identifier_units,
    }

    @staticmethod
    def register_identifier_check(func):
        """
        Register a function for checking identifiers.

        Parameters
        ----------
        func : callable
            The function has to receive a single argument, the name of the
            identifier to check, and raise a ValueError if the identifier
            violates any rule.

        """
        if not callable(func):
            raise ValueError("Can only register callables.")

        Equations.identifier_checks.add(func)

    @staticmethod
    def check_identifier(identifier):
        """
        Perform all the registered checks. Checks can be registered via
        `Equations.register_identifier_check`.

        Parameters
        ----------
        identifier : str
            The identifier that should be checked

        Raises
        ------
        ValueError
            If any of the registered checks fails.
        """
        for check_func in Equations.identifier_checks:
            check_func(identifier)

    def check_identifiers(self):
        """
        Check all identifiers for conformity with the rules.

        Raises
        ------
        ValueError
            If an identifier does not conform to the rules.

        See also
        --------
        Equations.check_identifier : The function that is called for each identifier.
        """
        for name in self.names:
            Equations.check_identifier(name)

    def get_substituted_expressions(self, variables=None, include_subexpressions=False):
        """
        Return a list of ``(varname, expr)`` tuples, containing all
        differential equations (and optionally subexpressions) with all the
        subexpression variables substituted with the respective expressions.

        Parameters
        ----------
        variables : dict, optional
            A mapping of variable names to `Variable`/`Function` objects.
        include_subexpressions : bool
            Whether also to return substituted subexpressions. Defaults to
            ``False``.

        Returns
        -------
        expr_tuples : list of (str, `CodeString`)
            A list of ``(varname, expr)`` tuples, where ``expr`` is a
            `CodeString` object with all subexpression variables substituted
            with the respective expression.
        """
        if self._substituted_expressions is None:
            self._substituted_expressions = []
            substitutions = {}
            for eq in self.ordered:
                # Skip parameters
                if eq.expr is None:
                    continue

                new_sympy_expr = str_to_sympy(eq.expr.code, variables).xreplace(
                    substitutions
                )
                new_str_expr = sympy_to_str(new_sympy_expr)
                expr = Expression(new_str_expr)

                if eq.type == SUBEXPRESSION:
                    if eq.var_type == INTEGER:
                        sympy_var = sympy.Symbol(eq.varname, integer=True)
                    else:
                        sympy_var = sympy.Symbol(eq.varname, real=True)
                    substitutions.update(
                        {sympy_var: str_to_sympy(expr.code, variables)}
                    )
                    self._substituted_expressions.append((eq.varname, expr))
                elif eq.type == DIFFERENTIAL_EQUATION:
                    #  a differential equation that we have to check
                    self._substituted_expressions.append((eq.varname, expr))
                else:
                    raise AssertionError(f"Unknown equation type {eq.type}")

        if include_subexpressions:
            return self._substituted_expressions
        else:
            return [
                (name, expr)
                for name, expr in self._substituted_expressions
                if self[name].type == DIFFERENTIAL_EQUATION
            ]

    def _get_stochastic_type(self):
        """
        Returns the type of stochastic differential equations (additivive or
        multiplicative). The system is only classified as ``additive`` if *all*
        equations have only additive noise (or no noise).

        Returns
        -------
        type : str
            Either ``None`` (no noise variables), ``'additive'`` (factors for
            all noise variables are independent of other state variables or
            time), ``'multiplicative'`` (at least one of the noise factors
            depends on other state variables and/or time).
        """

        if not self.is_stochastic:
            return None

        for _, expr in self.get_substituted_expressions():
            _, stochastic = expr.split_stochastic()
            if stochastic is not None:
                for factor in stochastic.values():
                    if "t" in factor.identifiers:
                        # noise factor depends on time
                        return "multiplicative"

                    for identifier in factor.identifiers:
                        if identifier in self.diff_eq_names:
                            # factor depends on another state variable
                            return "multiplicative"

        return "additive"

    ############################################################################
    # Properties
    ############################################################################

    # Lists of equations or (variable, expression tuples)
    ordered = property(
        lambda self: sorted(
            self._equations.values(), key=lambda key: (key.update_order, key.varname)
        ),
        doc=(
            "A list of all equations, sorted "
            "according to the order in which they should "
            "be updated"
        ),
    )

    diff_eq_expressions = property(
        lambda self: [
            (varname, eq.expr)
            for varname, eq in self.items()
            if eq.type == DIFFERENTIAL_EQUATION
        ],
        doc=(
            "A list of (variable name, expression) "
            "tuples of all differential equations."
        ),
    )

    eq_expressions = property(
        lambda self: [
            (varname, eq.expr)
            for varname, eq in self.items()
            if eq.type in (SUBEXPRESSION, DIFFERENTIAL_EQUATION)
        ],
        doc="A list of (variable name, expression) tuples of all equations.",
    )

    # Sets of names

    names = property(
        lambda self: {eq.varname for eq in self.ordered},
        doc="All variable names defined in the equations.",
    )

    diff_eq_names = property(
        lambda self: {
            eq.varname for eq in self.ordered if eq.type == DIFFERENTIAL_EQUATION
        },
        doc="All differential equation names.",
    )

    subexpr_names = property(
        lambda self: {eq.varname for eq in self.ordered if eq.type == SUBEXPRESSION},
        doc="All subexpression names.",
    )

    eq_names = property(
        lambda self: {
            eq.varname
            for eq in self.ordered
            if eq.type in (DIFFERENTIAL_EQUATION, SUBEXPRESSION)
        },
        doc="All equation names (including subexpressions).",
    )

    parameter_names = property(
        lambda self: {eq.varname for eq in self.ordered if eq.type == PARAMETER},
        doc="All parameter names.",
    )

    dimensions = property(
        lambda self: {var: eq.dim for var, eq in self._equations.items()},
        doc=(
            "Dictionary of all internal variables and their "
            "corresponding physical dimensions."
        ),
    )

    identifiers = property(
        lambda self: set().union(*[eq.identifiers for eq in self._equations.values()])
        - self.names,
        doc=(
            "Set of all identifiers used in the equations, "
            "excluding the variables defined in the equations"
        ),
    )

    stochastic_variables = property(
        lambda self: {
            variable
            for variable in self.identifiers
            if variable == "xi" or variable.startswith("xi_")
        }
    )

    # general properties
    is_stochastic = property(
        lambda self: len(self.stochastic_variables) > 0,
        doc="Whether the equations are stochastic.",
    )

    stochastic_type = property(fget=_get_stochastic_type)

    def _sort_subexpressions(self):
        """
        Sorts the subexpressions in a way that resolves their dependencies
        upon each other. After this method has been run, the subexpressions
        returned by the ``ordered`` property are in the order in which
        they should be updated
        """

        # Get a dictionary of all the dependencies on other subexpressions,
        # i.e. ignore dependencies on parameters and differential equations
        static_deps = {}
        for eq in self._equations.values():
            if eq.type == SUBEXPRESSION:
                static_deps[eq.varname] = [
                    dep
                    for dep in eq.identifiers
                    if dep in self._equations
                    and self._equations[dep].type == SUBEXPRESSION
                ]

        try:
            sorted_eqs = topsort(static_deps)
        except ValueError:
            raise ValueError(
                "Cannot resolve dependencies between static "
                "equations, dependencies contain a cycle."
            )

        # put the equations objects in the correct order
        for order, static_variable in enumerate(sorted_eqs):
            self._equations[static_variable].update_order = order

        # Sort differential equations and parameters after subexpressions
        for eq in self._equations.values():
            if eq.type == DIFFERENTIAL_EQUATION:
                eq.update_order = len(sorted_eqs)
            elif eq.type == PARAMETER:
                eq.update_order = len(sorted_eqs) + 1

    @property
    def dependencies(self):
        """
        Calculate the dependencies of all differential equations and
        subexpressions.
        """
        # Create a dictionary mapping differential equations and
        # subexpressions to a list of their dependencies within the equations
        # (ignoring external constants, unit names, etc.)
        # Note that a differential equation such as "dv/dt = -v / tau" does not
        # mean that the variable "v" depends on itself. To make the distinction between
        # a variable and its derivative, we use the variable name + the prime symbol
        # in this dictionary.
        # As an example, the equations:
        #   dv/dt = I_m / C_m : volt
        #   I_m = I_ext + I_pas : amp
        #   I_ext = 1*nA + sin(2*pi*100*Hz*t)*nA : amp
        #   I_pas = g_L*(E_L - v) : amp
        # would be translated into the following dictionary
        #  {"v" : [],
        #   "v'": ["I_m"]
        #   "I_m": ["I_ext", "I_pas"],
        #   "I_ext": [],
        #   "I_pas": ["v"] }
        deps = {}
        for eq in self._equations.values():
            if eq.type == SUBEXPRESSION:
                name = eq.varname
            elif eq.type == DIFFERENTIAL_EQUATION:
                name = eq.varname + "'"
                deps[eq.varname] = []
            else:
                continue
            deps[name] = [
                dep
                for dep in eq.identifiers
                if dep in self._equations and self._equations[dep].type != PARAMETER
            ]
        try:
            sorted_eqs = topsort(deps)
        except ValueError:
            raise ValueError(
                "Cannot resolve dependencies between static "
                "equations, dependencies contain a cycle."
            )
        # Remove the dummy entries for differential equations and rename
        # x' → x
        sorted_eqs = [
            x.replace("'", "") for x in sorted_eqs if x not in self.diff_eq_names
        ]
        # Now recursively fill in the dependencies – this only needs a single
        # pass due to the previous sorting
        deps = {}
        Dependency = namedtuple(
            "Dependency", ["equation", "via"], defaults=((),)
        )  # default for via is empty tuple
        for eq in sorted_eqs:
            dep_names = {
                dep for dep in self._equations[eq].identifiers if dep in self._equations
            }
            deps[eq] = [Dependency(equation=self._equations[dep]) for dep in dep_names]
            # add all indirect dependencies
            for dep in dep_names:
                for indirect_dep in deps.get(dep, []):
                    if indirect_dep.equation.varname == dep:
                        continue  # do not go into recursion if a variable depends on itself
                    if any(
                        indirect_dep.equation.varname == existing_dep.equation.varname
                        for existing_dep in deps[eq]
                    ):
                        continue  # Do not add indirect dependencies for things we also depend on directly
                    deps[eq].append(
                        Dependency(
                            equation=indirect_dep.equation,
                            via=(dep,) + indirect_dep.via,
                        )
                    )
        return deps

    def check_units(self, group, run_namespace):
        """
        Check all the units for consistency.

        Parameters
        ----------
        group : `Group`
            The group providing the context
        run_namespace : dict-like, optional
            An additional namespace that is used for variable lookup (if not
            defined, the implicit namespace of local variables is used).
        level : int, optional
            How much further to go up in the stack to find the calling frame

        Raises
        ------
        DimensionMismatchError
            In case of any inconsistencies.
        """
        all_variables = dict(group.variables)
        external = frozenset().union(
            *[expr.identifiers for _, expr in self.eq_expressions]
        )
        external -= set(all_variables.keys())

        resolved_namespace = group.resolve_all(
            external, run_namespace, user_identifiers=external
        )  # all variables are user defined

        all_variables.update(resolved_namespace)
        for var, eq in self._equations.items():
            if eq.type == PARAMETER:
                # no need to check units for parameters
                continue

            if eq.type == DIFFERENTIAL_EQUATION:
                try:
                    check_dimensions(
                        str(eq.expr), self.dimensions[var] / second.dim, all_variables
                    )
                except DimensionMismatchError as ex:
                    raise DimensionMismatchError(
                        "Inconsistent units in "
                        "differential equation "
                        f"defining variable '{eq.varname}':"
                        f"\n{ex.desc}",
                        *ex.dims,
                    ) from ex
            elif eq.type == SUBEXPRESSION:
                try:
                    check_dimensions(str(eq.expr), self.dimensions[var], all_variables)
                except DimensionMismatchError as ex:
                    raise DimensionMismatchError(
                        "Inconsistent units in "
                        f"subexpression {eq.varname}:"
                        f"\n%{ex.desc}",
                        *ex.dims,
                    ) from ex
            else:
                raise AssertionError(f"Unknown equation type: '{eq.type}'")

    def check_flags(self, allowed_flags, incompatible_flags=None):
        """
        Check the list of flags.

        Parameters
        ----------
        allowed_flags : dict
             A dictionary mapping equation types (PARAMETER,
             DIFFERENTIAL_EQUATION, SUBEXPRESSION) to a list of strings (the
             allowed flags for that equation type)
        incompatible_flags : list of tuple
            A list of flag combinations that are not allowed for the same
            equation.
        Notes
        -----
        Not specifying allowed flags for an equation type is the same as
        specifying an empty list for it.

        Raises
        ------
        ValueError
            If any flags are used that are not allowed.
        """
        if incompatible_flags is None:
            incompatible_flags = []

        for eq in self.values():
            for flag in eq.flags:
                if eq.type not in allowed_flags or len(allowed_flags[eq.type]) == 0:
                    raise ValueError(
                        f"Equations of type '{eq.type}' cannot have any flags."
                    )
                if flag not in allowed_flags[eq.type]:
                    raise ValueError(
                        f"Equations of type '{eq.type}' cannot have a "
                        f"flag '{flag}', only the following flags "
                        f"are allowed: {allowed_flags[eq.type]}"
                    )
                # Check for incompatibilities
                for flag_combinations in incompatible_flags:
                    if flag in flag_combinations:
                        remaining_flags = set(flag_combinations) - {flag}
                        for remaining_flag in remaining_flags:
                            if remaining_flag in eq.flags:
                                raise ValueError(
                                    f"Flag '{flag}' cannot be "
                                    "combined with flag "
                                    f"'{remaining_flag}'"
                                )

    ############################################################################
    # Representation
    ############################################################################

    def __str__(self):
        strings = [str(eq) for eq in self.ordered]
        return "\n".join(strings)

    def __repr__(self):
        return f"<Equations object consisting of {len(self._equations)} equations>"

    def _latex(self, *args):
        equations = []
        for eq in self._equations.values():
            # do not use SingleEquations._latex here as we want nice alignment
            varname = sympy.Symbol(eq.varname)
            if eq.type == DIFFERENTIAL_EQUATION:
                lhs = r"\frac{\mathrm{d}" + sympy.latex(varname) + r"}{\mathrm{d}t}"
            else:
                # Normal equation or parameter
                lhs = varname
            if not eq.type == PARAMETER:
                rhs = str_to_sympy(eq.expr.code)
            if len(eq.flags):
                flag_str = ", flags: " + ", ".join(eq.flags)
            else:
                flag_str = ""
            if eq.type == PARAMETER:
                eq_latex = r"{} &&& \text{{(unit: ${}${})}}".format(
                    sympy.latex(lhs),
                    sympy.latex(get_unit(eq.dim)),
                    flag_str,
                )
            else:
                eq_latex = r"{} &= {} && \text{{(unit of ${}$: ${}${})}}".format(
                    lhs,  # already a string
                    sympy.latex(rhs),
                    sympy.latex(varname),
                    sympy.latex(get_unit(eq.dim)),
                    flag_str,
                )
            equations.append(eq_latex)
        return r"\begin{align*}" + (r"\\" + "\n").join(equations) + r"\end{align*}"

    def _repr_latex_(self):
        return sympy.latex(self)

    def _repr_pretty_(self, p, cycle):
        """Pretty printing for ipython"""
        if cycle:
            # Should never happen
            raise AssertionError("Cyclical call of 'Equations._repr_pretty_'")
        for eq in self._equations.values():
            p.pretty(eq)
            p.breakable("\n")


def is_stateful(expression, variables):
    """
    Whether the given expression refers to stateful functions (and is therefore
    not guaranteed to give the same result if called repetively).

    Parameters
    ----------
    expression : `sympy.Expression`
        The sympy expression to check.
    variables : dict
        The dictionary mapping variable names to `Variable` or `Function`
        objects.

    Returns
    -------
    stateful : bool
        ``True``, if the given expression refers to a stateful function like
        ``rand()`` and ``False`` otherwise.
    """
    func_name = str(expression.func)
    func_variable = variables.get(func_name, None)
    if func_variable is not None and not func_variable.stateless:
        return True
    for arg in expression.args:
        if is_stateful(arg, variables):
            return True
    return False


def check_subexpressions(group, equations, run_namespace):
    """
    Checks the subexpressions in the equations and raises an error if a
    subexpression refers to stateful functions without being marked as
    "constant over dt".

    Parameters
    ----------
    group : `Group`
        The group providing the context.
    equations : `Equations`
        The equations to check.
    run_namespace : dict
        The run namespace for resolving variables.

    Raises
    ------
    SyntaxError
        For subexpressions not marked as "constant over dt" that refer to
        stateful functions.
    """
    for eq in equations.ordered:
        if eq.type == SUBEXPRESSION:
            # Check whether the expression is stateful (most commonly by
            # referring to rand() or randn()
            variables = group.resolve_all(
                eq.identifiers,
                run_namespace,
                # we don't need to raise any warnings
                # for the user here, warnings will
                # be raised in create_runner_codeobj
                user_identifiers=set(),
            )
            expression = str_to_sympy(eq.expr.code, variables=variables)

            # Check whether the expression refers to stateful functions
            if is_stateful(expression, variables):
                raise SyntaxError(
                    f"The subexpression '{eq.varname}' refers to a "
                    "stateful function (e.g. rand()). Such "
                    "expressions should only be evaluated "
                    "once per timestep, add the 'constant "
                    "over dt' flag."
                )


def extract_constant_subexpressions(eqs):
    without_const_subexpressions = []
    const_subexpressions = []
    for eq in eqs.ordered:
        if eq.type == SUBEXPRESSION and "constant over dt" in eq.flags:
            flags = set(eq.flags) - {"constant over dt"}
            without_const_subexpressions.append(
                SingleEquation(
                    PARAMETER, eq.varname, eq.dim, var_type=eq.var_type, flags=flags
                )
            )
            const_subexpressions.append(eq)
        else:
            without_const_subexpressions.append(eq)

    return (Equations(without_const_subexpressions), Equations(const_subexpressions))
