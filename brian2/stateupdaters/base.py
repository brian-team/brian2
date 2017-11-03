'''
This module defines the `StateUpdateMethod` class that acts as a base class for
all stateupdaters and allows to register stateupdaters so that it is able to
return a suitable stateupdater object for a given set of equations. This is used
for example in `NeuronGroup` when no state updater is given explicitly.
'''
from abc import abstractmethod, ABCMeta
import collections
import time

from brian2.utils.caching import cached
from brian2.utils.logger import get_logger

__all__ = ['StateUpdateMethod']

logger = get_logger(__name__)

class UnsupportedEquationsException(Exception):
    pass


def extract_method_options(method_options, default_options):
    '''
    Helper function to check ``method_options`` against options understood by
    this state updater, and setting default values for all unspecified options.

    Parameters
    ----------
    method_options : dict or None
        The options that the user specified for the state update.
    default_options : dict
        The default option values for this state updater (each admissible option
        needs to be present in this dictionary). To specify that a state updater
        does not take any options, provide an empty dictionary as the argument.

    Returns
    -------
    options : dict
        The final dictionary with all the options either at their default or at
        the user-specified value.

    Raises
    ------
    KeyError
        If the user specifies an option that is not understood by this state
        updater.

    Examples
    --------
    >>> options = extract_method_options({'a': True}, default_options={'b': False, 'c': False})  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    KeyError: 'method_options specifies "a", but this is not an option for this state updater. Avalaible options are: "b", "c".'
    >>> options = extract_method_options({'a': True}, default_options={})  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    KeyError: 'method_options specifies "a", but this is not an option for this state updater. This state updater does not accept any options.'
    >>> options = extract_method_options({'a': True}, default_options={'a': False, 'b': False})
    >>> sorted(options.items())
    [('a', True), ('b', False)]
    '''
    if method_options is None:
        method_options = {}
    for key in method_options:
        if key not in default_options:
            if len(default_options):
                keys = sorted(default_options.iterkeys())
                options = ('Avalaible options are: ' +
                           ', '.join('"%s"' % key for key in keys) + '.')
            else:
                options = 'This state updater does not accept any options.'
            raise KeyError('method_options specifies "{key}", but this '
                           'is not an option for this state updater. '
                           '{options}'.format(key=key, options=options))
    filled_options = dict(default_options)
    filled_options.update(method_options)
    return filled_options


class StateUpdateMethod(object):
    __metaclass__ = ABCMeta

    #: A dictionary mapping state updater names to `StateUpdateMethod` objects
    stateupdaters = dict()

    @abstractmethod
    def __call__(self, equations, variables=None, method_options=None):
        '''
        Generate abstract code from equations. The method also gets the
        the variables because some state updaters have to check whether
        variable names reflect other state variables (which can change from
        timestep to timestep) or are external values (which stay constant during
        a run)  For convenience, this arguments are optional -- this allows to
        directly see what code a state updater generates for a set of equations
        by simply writing ``euler(eqs)``, for example.
        
        Parameters
        ----------
        equations : `Equations`
            The model equations.
        variables : dict, optional
            The `Variable` objects for the model variables.
        method_options : dict, optional
            Additional options specific to the state updater.
        Returns
        -------
        code : str
            The abstract code performing a state update step.
        '''
        pass

    @staticmethod
    def register(name, stateupdater):
        '''
        Register a state updater. Registered state updaters can be referred to
        via their name.
        
        Parameters
        ----------
        name : str
            A short name for the state updater (e.g. `'euler'`)
        stateupdater : `StateUpdaterMethod`
            The state updater object, e.g. an `ExplicitStateUpdater`.
        '''
        
        # only deal with lower case names -- we don't want to have 'Euler' and
        # 'euler', for example
        name = name.lower()
        if name in StateUpdateMethod.stateupdaters:
            raise ValueError(('A stateupdater with the name "%s" '
                              'has already been registered') % name)

        if not isinstance(stateupdater, StateUpdateMethod):
            raise ValueError(('Given stateupdater of type %s does not seem to '
                              'be a valid stateupdater.' % str(type(stateupdater))))

        StateUpdateMethod.stateupdaters[name] = stateupdater

    @staticmethod
    @cached
    def apply_stateupdater(equations, variables, method, method_options=None, group_name=None):
        '''
        apply_stateupdater(equations, variables, method, method_options=None, group_name=None)

        Applies a given state updater to equations. If a `method` is given, the
        state updater with the given name is used or if is a callable, then it
        is used directly. If a `method` is a list of names, all the
        methods will be tried until one that doesn't raise an
        `UnsupportedEquationsException` is found.
        
        Parameters
        ----------
        equations : `Equations`
            The model equations.
        variables : `dict`
            The dictionary of `Variable` objects, describing the internal
            model variables.
        method : {callable, str, list of str}
            A callable usable as a state updater, the name of a registered
            state updater or a list of names of state updaters.

        Returns
        -------
        abstract_code : str
            The code integrating the given equations.
        '''
        if (isinstance(method, collections.Iterable) and
                not isinstance(method, basestring)):
            the_method = None
            start_time = time.time()
            for one_method in method:
                try:
                    one_method_start_time = time.time()
                    code = StateUpdateMethod.apply_stateupdater(equations,
                                                                variables,
                                                                one_method,
                                                                group_name=group_name)
                    the_method = one_method
                    one_method_time = time.time() - one_method_start_time
                    break
                except UnsupportedEquationsException:
                    pass
                except TypeError:
                    raise TypeError(('Each element in the list of methods has '
                                     'to be a string or a callable, got %s.')
                                    % type(one_method))
            total_time = time.time() - start_time
            if the_method is None:
                raise ValueError(('No stateupdater that is suitable for the '
                                  'given equations has been found.'))

            # If only one method was tried
            if method[0] == the_method:
                timing = 'took %.2fs' % one_method_time
            else:
                timing = ('took %.2fs, trying other methods took '
                          '%.2fs') % (one_method_time,
                                      total_time-one_method_time)

            if group_name is not None:
                msg_text = ("No numerical integration method specified for group "
                            "'{group_name}', using method '{method}' ({timing}).")
            else:
                msg_text = ("No numerical integration method specified, "
                            "using method '{method}' ({timing}).")
            logger.info(msg_text.format(group_name=group_name,
                                        method=the_method,
                                        timing=timing), 'method_choice')
        else:
            if hasattr(method, '__call__'):
                # if this is a standard state updater, i.e. if it has a
                # can_integrate method, check this method and raise a warning if it
                # claims not to be applicable.
                stateupdater = method
                method = getattr(stateupdater, '__name__', repr(stateupdater))  # For logging, get a nicer name
            elif isinstance(method, basestring):
                method = method.lower()  # normalize name to lower case
                stateupdater = StateUpdateMethod.stateupdaters.get(method, None)
                if stateupdater is None:
                    raise ValueError('No state updater with the name "%s" '
                                     'is known' % method)
            else:
                raise TypeError(('method argument has to be a string, a '
                                 'callable, or an iterable of such objects. '
                                 'Got %s') % type(method))
            start_time = time.time()
            code = stateupdater(equations, variables, method_options)
            method_time = time.time() - start_time
            timing = 'took %.2fs' % method_time
            if group_name is not None:
                logger.debug(('Group {group_name}: using numerical integration '
                             'method {method} ({timing})').format(group_name=group_name,
                                                                  method=method,
                                                                  timing=timing),
                             'method_choice')
            else:
                logger.debug(('Using numerical integration method: {method} '
                              '({timing})').format(method=method,
                                                   timing=timing),
                             'method_choice')

        return code
