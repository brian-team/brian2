'''
TODO: BrianObject covers what was previously covered by magic.InstanceTracker
and NetworkOperation
'''

__all__ = ['BrianObject']

class BrianObject(object):
    '''
    TODO
    '''
    def __init__(self, when='start'):
        if not isinstance(when, str):
            raise TypeError("when attribute should be a string, was "+repr(when))
        self.when = when
        
    def update(self):
        '''
        All BrianObjects should define an update() method.
        '''
        raise NotImplementedError
