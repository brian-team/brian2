'''
Used to dynamically switch classes and functions at runtime

For usage examples, see the bottom of this module.
'''

__all__ = ['class_switcher', 'SwitchableClass']

class DynamicSwitcher(object):
    '''
    Singleton class that is used to store mappings ``orig->new`` for original and new classes/functions/methods.
    '''
    def __init__(self):
        self.mappings = {}
    
    def __setitem__(self, item, val):
        if hasattr(item, 'im_func'):
            newitem = item.im_func
        else:
            newitem = item
        self.mappings[id(newitem)] = val
    
    def __getitem__(self, item):
        if hasattr(item, 'im_func'):
            newitem = item.im_func
        else:
            newitem = item
        return self.mappings.get(id(newitem), item)
    
    def get(self, item, default):
        if hasattr(item, 'im_func'):
            newitem = item.im_func
        else:
            newitem = item
        return self.mappings.get(id(newitem), default)
    
    def reset(self):
        self.mappings.clear()
    
#: Single instance of ``DynamicSwitcher``
switcher = DynamicSwitcher()

class SwitchableClass(object):
    '''
    Derive from this class to make the class dynamically switchable
    '''
    def __new__(cls, *args, **kwds):
        newcls = switcher[cls]
        return object.__new__(newcls, *args, **kwds)

def switchable_function(f):
    '''
    Decorator for functions and methods to make them dynamically switchable.
    '''
    def switched_f(*args, **kwds):
        return switcher.get(switched_f, f)(*args, **kwds)
    return switched_f

if __name__=='__main__':
    
    class X(SwitchableClass):
        def __init__(self):
            print 'X.__init__'
        def f(self):
            print 'X.f'
        @switchable_function
        def h(self):
            print 'X.h'
    
    class Y(X):
        def __init__(self):
            print 'Y.__init__'
        def f(self):
            print 'Y.f'
            
    class Z(X):
        def __init__(self):
            X.__init__(self)
            print 'Z.__init__'
        def f(self):
            print 'Z.f'

    @switchable_function
    def g():
        print 'g'
    
    def new_g():
        print 'new_g'
        
    def new_h(self):
        print 'new_h'
    
    x = Z()
    
    print
        
    x.f()
    g()
    x.h()
    
    print
    
    switcher[X] = Y
    switcher[g] = new_g
    switcher[X.h] = new_h
    
    x.f()
    g()
    x.h()
    