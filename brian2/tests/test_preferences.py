from brian2 import *
from numpy.testing import assert_raises, assert_equal
from nose import with_setup

@with_setup(teardown=restore_initial_state)
def test_preferences():
    assert_raises(KeyError, lambda: brian_prefs.fake_nonexisting_pref)
    brian_prefs.define('fake_weave_compiler', 'gcc',
                       'The distutils compiler specifier for weave compilation.')
    brian_prefs.fake_weave_compiler
    brian_prefs.fake_weave_compiler = 'msvc'
    brian_prefs.fake_weave_compiler
    assert_raises(TypeError, lambda: setattr(brian_prefs, 'fake_weave_compiler', 5))
    brian_prefs.define('fake_pref_with_units', 1*second, 'Fake pref with units')
    brian_prefs.fake_pref_with_units = 2*second
    assert_raises(DimensionMismatchError,
                  lambda: setattr(brian_prefs, 'fake_pref_with_units', 1*amp))
    brian_prefs.define('fake_weave_compiler_options', ['-O3', '-ffast-math'],
                       'The extra compiler options for weave compilation.')
    brian_prefs.documentation

if __name__=='__main__':
    test_preferences()
    print brian_prefs.documentation
