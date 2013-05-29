from numpy import float64, float32
from StringIO import StringIO

from brian2 import restore_initial_state, volt, amp
from brian2.core.preferences import (DefaultValidator, BrianPreference,
                                     BrianGlobalPreferences, PreferenceError,
                                     BrianGlobalPreferencesView)

from numpy.testing import assert_equal, assert_raises
from nose import with_setup

@with_setup(teardown=restore_initial_state)
def test_defaultvalidator():
    # Test that the default validator checks the class
    validator = DefaultValidator(5)
    assert validator(3)
    assert not validator('3')
    validator = DefaultValidator('astring')
    assert validator('another')
    assert not validator(3)
    # test that the default validator checks the units
    validator = DefaultValidator(3*volt)
    assert validator(2*volt)
    assert not validator(1*amp)


@with_setup(teardown=restore_initial_state)
def test_brianpreference():
    # check default args
    pref = BrianPreference(1./3, 'docs')
    assert not pref.validator(1)
    assert pref.docs=='docs'
    assert pref.default==1./3
    assert pref.representor(pref.default)==repr(1./3)

@with_setup(teardown=restore_initial_state)
def test_preference_name_checking():
    '''
    Test that you cannot set illegal preference names.
    '''
    gp = BrianGlobalPreferences()
    
    # Name that starts with an underscore
    assert_raises(PreferenceError, lambda: gp.register_preferences('dummy', 'dummy doc',
                                                                   _notalegalname=BrianPreference(True, 'some preference')
                                                                   ))

    # Name that clashes with a method name
    assert_raises(PreferenceError, lambda: gp.register_preferences('dummy', 'dummy doc',
                                                                   update=BrianPreference(True, 'some preference')
                                                                   ))
                   
    gp.register_preferences('a', 'dummy doc',
                            b=BrianPreference(True, 'some preference'))
    
    #Trying to register a subcategory that would shadow a preference
    assert_raises(PreferenceError, lambda: gp.register_preferences('a.b', 'dummy doc',
                                                                   name=BrianPreference(True, 'some preference')
                                                                   ))


    gp.register_preferences('b.c', 'dummy doc',
                            name=BrianPreference(True, 'some preference'))
        
    #Trying to register a preference that clashes with an existing category
    assert_raises(PreferenceError, lambda: gp.register_preferences('b', 'dummy doc',
                                                                   c=BrianPreference(True, 'some preference')
                                                                   ))


@with_setup(teardown=restore_initial_state)
def test_brianglobalpreferences():
    # test that pre-setting a nonexistent preference in a subsequently
    # existing base name raises an error at the correct point
    gp = BrianGlobalPreferences()
    
    # This shouldn't work, in user code only registered preferences can be set
    assert_raises(PreferenceError, lambda: gp.__setitem__('a.b', 5))
    
    # This uses the method that is used when reading preferences from a file
    gp._set_preference('a.b', 5)
    gp._set_preference('a.c', 5)
    assert_raises(PreferenceError, gp.register_preferences, 'a', 'docs for a',
                  b=BrianPreference(5, 'docs for b'))
    # test that post-setting a nonexistent preference in an existing base
    # name raises an error
    gp = BrianGlobalPreferences()
    gp.register_preferences('a', 'docs for a',
                            b=BrianPreference(5, 'docs for b'))
    assert_raises(PreferenceError, gp.__setitem__, 'a.c', 5)
    # Test pre and post-setting some correct names but valid and invalid values
    gp = BrianGlobalPreferences()
    gp._set_preference('a.b', 5)
    gp.register_preferences('a', 'docs for a',
        b=BrianPreference(5, 'docs for b'),
        c=BrianPreference(1*volt, 'docs for c'),
        d=BrianPreference(0, 'docs for d', validator=lambda x:x>=0),
        e=BrianPreference(float64, 'docs for e',
                          representor=lambda x: x.__name__),
        )
    assert gp['a.c']==1*volt
    gp['a.c'] = 2*volt
    assert_raises(PreferenceError, gp.__setitem__, 'a.c', 3*amp)
    gp['a.d'] = 2.0
    assert_raises(PreferenceError, gp.__setitem__, 'a.d', -1)
    gp['a.e'] = float32
    assert_raises(PreferenceError, gp.__setitem__, 'a.e', 0)
    # test backup and restore
    gp._backup()
    gp['a.d'] = 10
    assert gp['a.d']==10
    gp._restore()
    assert gp['a.d']==2.0
    # test that documentation and as_file generation runs without error, but
    # don't test for values because we might change the organisation of it
    assert len(gp.get_documentation())
    gp.as_file
    gp.defaults_as_file
    # test that reading a preference file works as expected
    pref_file = StringIO('''
        # a comment
        a.b = 10
        [a]
        c = 5*volt
        d = 1
        e = float64
        ''')
    gp.read_preference_file(pref_file)
    assert gp['a.b']==10
    assert gp['a.c']==5*volt
    assert gp['a.d']==1
    assert gp['a.e']==float64
    # test that reading a badly formatted prefs file fails
    pref_file = StringIO('''
        [a
        b = 10
        ''')
    assert_raises(PreferenceError, gp.read_preference_file, pref_file)
    # test that reading a well formatted prefs file with an invalid value fails
    pref_file = StringIO('''
        a.b = 'oh no, not a string'
        ''')
    assert_raises(PreferenceError, gp.read_preference_file, pref_file)
    # assert that writing the prefs to a file and loading them gives the
    # same values
    gp = BrianGlobalPreferences()
    gp.register_preferences('a', 'docs for a',
        b=BrianPreference(5, 'docs for b'),
        )
    gp._backup()
    gp['a.b'] = 10
    str_modified = gp.as_file
    str_defaults = gp.defaults_as_file
    gp['a.b'] = 15
    gp.read_preference_file(StringIO(str_modified))
    assert gp['a.b']==10
    gp.read_preference_file(StringIO(str_defaults))
    assert gp['a.b']==5
    # check that load_preferences works, but nothing about its values
    gp = BrianGlobalPreferences()
    gp.load_preferences()

@with_setup(teardown=restore_initial_state)
def test_preference_name_access():
    '''
    Test various ways of accessing preferences
    '''
    
    gp = BrianGlobalPreferences()
    
    gp.register_preferences('main', 'main category',
                            name=BrianPreference(True, 'some preference'))    
    gp.register_preferences('main.sub', 'subcategory',
                            name2=BrianPreference(True, 'some preference'))
    
    # Keyword based access
    assert gp['main.name']
    assert gp['main.sub.name2']
    gp['main.name'] = False
    gp['main.sub.name2'] = False
    
    # Attribute based access
    assert not gp.main.name  # we set it to False above
    assert not gp.main.sub.name2
    gp.main.name = True
    gp.main.sub.name2 = True
    
    # Mixed access
    assert gp.main['name']
    assert gp['main'].name
    assert gp.main['sub'].name2
    assert gp['main'].sub['name2']
    
    # Accessing categories
    assert isinstance(gp['main'], BrianGlobalPreferencesView)
    assert isinstance(gp['main.sub'], BrianGlobalPreferencesView)
    assert isinstance(gp.main, BrianGlobalPreferencesView)
    assert isinstance(gp.main.sub, BrianGlobalPreferencesView)
    
    # Setting categories shouldn't work
    assert_raises(PreferenceError, lambda: gp.__setitem__('main', None))
    assert_raises(PreferenceError, lambda: gp.__setattr__('main', None))
    assert_raises(PreferenceError, lambda: gp.main.__setitem__('sub', None))
    assert_raises(PreferenceError, lambda: gp.main.__setattr__('sub', None))
    
    # Neither should deleting categories or preferences
    assert_raises(PreferenceError, lambda: gp.__delitem__('main'))
    assert_raises(PreferenceError, lambda: gp.__delattr__('main'))
    assert_raises(PreferenceError, lambda: gp.main.__delitem__('name'))
    assert_raises(PreferenceError, lambda: gp.main.__delattr__('name'))
    assert_raises(PreferenceError, lambda: gp.main.__delitem__('sub'))
    assert_raises(PreferenceError, lambda: gp.main.__delattr__('sub'))
    
    #Errors for accessing non-existing preferences
    assert_raises(KeyError, lambda: gp['main.doesnotexist'])
    assert_raises(KeyError, lambda: gp['nonexisting.name'])
    assert_raises(KeyError, lambda: gp.main.doesnotexist)
    assert_raises(KeyError, lambda: gp.nonexisting.name)

    # Check dictionary functionality
    for name, value in gp.iteritems():
        assert gp[name] == value
    
    for name, value in gp.main.iteritems():
        assert gp.main[name] == value
    
    assert len(gp) == 2  # two preferences in total
    assert len(gp['main']) == 2  # both preferences are in the main category
    assert len(gp['main.sub']) == 1  # one preference in main.sub
    
    assert 'main.name' in gp
    assert 'name' in gp['main']
    assert 'name2' in gp['main.sub']
    assert not 'name' in gp['main.sub']
    
    gp['main.name'] = True
    gp.update({'main.name': False})
    assert not gp['main.name']
    
    gp.main.update({'name': True})
    assert gp['main.name']
    
    # Class based functionality
    assert 'main' in dir(gp)
    assert 'sub' in dir(gp.main)
    assert 'name' in dir(gp.main)

    # Check that the fiddling with getattr and setattr did not destroy the
    # access to standard attributes
    assert len(gp.prefs)
    assert gp.main._basename == 'main'


@with_setup(teardown=restore_initial_state)
def test_str_repr():
    # Just test whether str and repr do not throw an error and return something
    gp = BrianGlobalPreferences()    
    gp.register_preferences('main', 'main category',
                            name=BrianPreference(True, 'some preference'))
    
    assert len(str(gp))
    assert len(repr(gp))
    assert len(str(gp.main))
    assert len(repr(gp.main))

   
if __name__=='__main__':
    for t in [test_defaultvalidator,
              test_brianpreference,
              test_brianglobalpreferences,
              test_preference_name_checking,
              test_preference_name_access
              ]:
        t()
        restore_initial_state()
