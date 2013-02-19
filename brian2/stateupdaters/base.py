'''
This module defines the `Stateupdater` object that acts as a base class for all
stateupdaters and allows to register stateupdaters so that it is able to
return a suitable stateupdater object for a given set of equations. This is used
for example in `NeuronGroup` when no state updater is given explicitly.
'''
from abc import abstractmethod, ABCMeta

from brian2.utils.logger import get_logger


__all__ = ['StateUpdater']

logger = get_logger(__name__)

class StateUpdater(object):    
    __metaclass__ = ABCMeta
    
    #: A dictionary of registered stateupdaters (using a short name as the key)
    stateupdaters = {}
    
    @abstractmethod
    def get_priority(self, equations, namespace, specifiers):
        '''
        Determine whether the state updater is a suitable choice. Should return
        0 if it is not appropriate (e.g. non-linear equations for a linear
        state updater) and a value > 0 if it is appropriate. The number acts as
        a priority, i.e. if more than one state updater is possible, the one
        with the highest value is chosen.
        
        Parameters
        ----------
        equations : `Equations`
            The model equations.
        namespace : dict
            The namespace resolving the external identifiers used in the
            equations.
        specifiers : dict
            The `Specifier` objects for the model variables.
        
        Returns
        -------
        priority : int
            0, if this state updater is not a possible choice. A value > 0
            otherwise.
        '''
        pass
    
    @abstractmethod
    def __call__(self, equations):
        '''
        Generate abstract code from equations.
        
        Parameters
        ----------
        equations : `Equations`
            The model equations.
        
        Returns
        -------
        code : str
            The abstract code performing a state update step.
        '''
        pass
    
    @staticmethod
    def register(name, stateupdater):
        if name in StateUpdater.stateupdaters:
            raise ValueError(('A stateupdater with the name "%s" '
                              'has already been registered') % name)
        
        if not isinstance(stateupdater, StateUpdater):
            raise ValueError(('Given stateupdater of type %s does not seem to '
                              'be a valid stateupdater.' % str(type(stateupdater))))
        
        StateUpdater.stateupdaters[name] = stateupdater
    
    @staticmethod
    def determine_stateupdater(self, equations, namespace,
                               specifiers, name=None):
        '''
        Determine a suitable state updater. If a `name` is given, the
        state updater with the given name is used (if it is suitable).
        Otherwise, the suitable state updater with the highest priority is used.
        ''' 
        if name is not None:
            try:
                stateupdater = self.stateupdaters[name]
            except KeyError:
                raise ValueError('No state updater with the name "%s" '
                                 'is known' % name)
            if stateupdater.get_priority(equations, namespace, specifiers) == 0:
                raise ValueError(('The state updater "%s" cannot be used for '
                                  'the given equations' % name))
            return stateupdater
        
        # determine the best suitable state updater
        priorities = [(name, updater.get_priority(equations,
                                              namespace,
                                              specifiers))
                         for name, updater in self.stateupdaters.iteritems()]
        priorities.sort(key=lambda elem: elem[1], reverse=True)
        
        # If the list is empty or the first (=best) priority is 0, we did not
        # find anything suitable
        if len(priorities) == 0 or priorities[0][1] == 0:
            raise ValueError(('No stateupdater that is suitable for the given '
                              'equations has been found'))
        
        # The first entry in the list is the stateupdater of our choice
        logger.info('Using stateupdater "%s"' % priorities[0][0])
        
        return self.stateupdaters[priorities[0][0]]
        
    
