from numpy import float64, float32
from StringIO import StringIO

from brian2 import restore_initial_state, volt, amp
from brian2.core.preferences import (DefaultValidator, BrianPreference,
                                     BrianGlobalPreferences, PreferenceError,
                                     )

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
def test_brianglobalpreferences():
    # test that pre-setting a nonexistent preference in a subsequently
    # existing base name raises an error at the correct point
    gp = BrianGlobalPreferences()
    gp['a.b'] = 5
    gp['a.c'] = 5
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
    gp['a.b'] = 5
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
    gp.documentation
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
    

if __name__=='__main__':
    for t in [test_defaultvalidator,
              test_brianpreference,
              test_brianglobalpreferences,
              ]:
        t()
        restore_initial_state()
