'''
This module defines the `StateUpdateMethod` class that acts as a base class for
all stateupdaters and allows to register stateupdaters so that it is able to
return a suitable stateupdater object for a given set of equations. This is used
for example in `NeuronGroup` when no state updater is given explicitly.
'''
from abc import abstractmethod, ABCMeta

from brian2.utils.logger import get_logger

__all__ = ['StateUpdateMethod']

logger = get_logger(__name__)

class StateUpdateMethod(object):
    __metaclass__ = ABCMeta

    #: A list of registered (name, stateupdater) pairs (in the order of priority)
    stateupdaters = []

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
    def register(name, stateupdater, index=None):
        '''
        Register a state updater. Registered state updaters will be considered
        when no state updater is explicitly given (e.g. in `NeuronGroup`) and
        can be referred to via their name.
        
        Parameters
        ----------
        name : str
            A short name for the state updater (e.g. `'euler'`)
        stateupdater : `StateUpdaterMethod`
            The state updater object, e.g. an `ExplicitStateUpdater`.
        index : int, optional
            Where in the list of state updaters the given state updater should
            be inserted. State updaters have a higher priority of being chosen
            automatically if they appear earlier in the list. If no `index` is
            given, the state updater will be inserted at the end of the list.
        '''
        
        # only deal with lower case names -- we don't want to have 'Euler' and
        # 'euler', for example
        name = name.lower() 
        for registered_name, _ in StateUpdateMethod.stateupdaters:
            if registered_name == name:
                raise ValueError(('A stateupdater with the name "%s" '
                                  'has already been registered') % name)

        if not isinstance(stateupdater, StateUpdateMethod):
            raise ValueError(('Given stateupdater of type %s does not seem to '
                              'be a valid stateupdater.' % str(type(stateupdater))))

        if not index is None:
            try:
                index = int(index)
            except (TypeError, ValueError):
                raise TypeError(('Index argument should be an integer, is '
                                 'of type %s instead.') % type(index))
            StateUpdateMethod.stateupdaters.insert(index, (name, stateupdater))
        else:
            StateUpdateMethod.stateupdaters.append((name, stateupdater))

    @staticmethod
    def determine_stateupdater(equations, variables, method=None):
        '''
        Determine a suitable state updater. If a `method` is given, the
        state updater with the given name is used. In case it is a callable, it
        will be used even if it is a state updater that claims it is not
        applicable. If it is a string, the state updater registered with that
        name will be used, but in this case an error will be raised if it
        claims not to be applicable. If no `method` is given explicitly, the
        suitable state updater with the highest priority is used.
        
        Parameters
        ----------
        equations : `Equations`
            The model equations.
        variables : `dict`
            The dictionary of `Variable` objects, describing the internal
            model variables.
        method : {callable, str, ``None``}, optional
            A callable usable as a state updater, the name of a registered
            state updater or ``None`` (the default) 
        '''
        if hasattr(method, '__call__'):
            # if this is a standard state updater, i.e. if it has a
            # can_integrate method, check this method and raise a warning if it
            # claims not to be applicable.
            try:
                priority = method.can_integrate(equations, variables)
                if priority == 0:
                    logger.warn(('The manually specified state updater '
                                 'claims that it does not support the given '
                                 'equations.'))
            except AttributeError:
                # No can_integrate method
                pass
            
            logger.info('Using manually specified state updater: %r' % method)
            return method
        
        if method is not None:
            method = method.lower()  # normalize name to lower case
            stateupdater = None
            for name, registered_stateupdater in StateUpdateMethod.stateupdaters:
                if name == method:
                    stateupdater = registered_stateupdater
                    break
            if stateupdater is None:
                raise ValueError('No state updater with the name "%s" '
                                 'is known' % method)
            if not stateupdater.can_integrate(equations, variables):
                raise ValueError(('The state updater "%s" cannot be used for '
                                  'the given equations' % method))
            return stateupdater

        # determine the best suitable state updater
        best_stateupdater = None
        for name, stateupdater in StateUpdateMethod.stateupdaters:
            try:
                if stateupdater.can_integrate(equations, variables):
                    best_stateupdater = (name, stateupdater)
                    break
            except KeyError:
                logger.debug(('It could not be determined whether state '
                             'updater "%s" is able to integrate the equations, '
                              'it appears the namespace is not yet complete.'
                              % name))

        # No suitable state updater has been found
        if best_stateupdater is None:
            raise ValueError(('No stateupdater that is suitable for the given '
                              'equations has been found'))

        name, stateupdater = best_stateupdater
        logger.info('Using stateupdater "%s"' % name)
        return stateupdater
