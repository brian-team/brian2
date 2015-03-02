'''
This module defines the `StateUpdateMethod` class that acts as a base class for
all stateupdaters and allows to register stateupdaters so that it is able to
return a suitable stateupdater object for a given set of equations. This is used
for example in `NeuronGroup` when no state updater is given explicitly.
'''
from abc import abstractmethod, ABCMeta
import collections

from brian2.utils.logger import get_logger

__all__ = ['StateUpdateMethod']

logger = get_logger(__name__)

class StateUpdateMethod(object):
    __metaclass__ = ABCMeta

    #: A dictionary mapping state updater names to `StateUpdateMethod` objects
    stateupdaters = dict()

    @abstractmethod
    def can_integrate(self, equations, variables):
        '''
        Determine whether the state updater is a suitable choice. Should return
        ``False`` if it is not appropriate (e.g. non-linear equations for a
        linear state updater) and a ``True`` if it is appropriate.
        
        Parameters
        ----------
        equations : `Equations`
            The model equations.
        variables : dict
            The `Variable` objects for the model variables.
        
        Returns
        -------
        ability : bool
            ``True`` if this state updater is able to integrate the given
            equations, ``False`` otherwise.
        '''
        pass

    @abstractmethod
    def __call__(self, equations, variables=None):
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
    def determine_stateupdater(equations, variables, method):
        '''
        Determine a suitable state updater. If a `method` is given, the
        state updater with the given name is used. In case it is a callable, it
        will be used even if it is a state updater that claims it is not
        applicable. If it is a string, the state updater registered with that
        name will be used, but in this case an error will be raised if it
        claims not to be applicable. If a `method` is a list of names, all the
        methods will be tried until one that can integrate the equations is
        found.
        
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
        '''
        if hasattr(method, '__call__'):
            # if this is a standard state updater, i.e. if it has a
            # can_integrate method, check this method and raise a warning if it
            # claims not to be applicable.
            try:
                if not method.can_integrate(equations, variables):
                    logger.warn(('The manually specified state updater '
                                 'claims that it does not support the given '
                                 'equations.'))
            except AttributeError:
                # No can_integrate method
                pass
            
            logger.info('Using manually specified state updater: %r' % method)
            return method
        elif isinstance(method, basestring):
            method = method.lower()  # normalize name to lower case
            stateupdater = StateUpdateMethod.stateupdaters.get(method, None)
            if stateupdater is None:
                raise ValueError('No state updater with the name "%s" '
                                 'is known' % method)
            if not stateupdater.can_integrate(equations, variables):
                raise ValueError(('The state updater "%s" cannot be used for '
                                  'the given equations' % method))
            return stateupdater
        elif isinstance(method, collections.Iterable):
            for name in method:
                if name not in StateUpdateMethod.stateupdaters:
                    logger.warn('No state updater with the name "%s" '
                                'is known' % name, 'unkown_stateupdater')
                else:
                    stateupdater = StateUpdateMethod.stateupdaters[name]
                    try:
                        if stateupdater.can_integrate(equations, variables):
                            logger.info('Using stateupdater "%s"' % name)
                            return stateupdater
                    except KeyError:
                        logger.debug(('It could not be determined whether state '
                                      'updater "%s" is able to integrate the equations, '
                                      'it appears the namespace is not yet complete.'
                                      % name))

            raise ValueError(('No stateupdater that is suitable for the given '
                              'equations has been found'))
