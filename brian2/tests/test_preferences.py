from brian2 import *
from numpy.testing import assert_raises, assert_equal

assert_raises(KeyError, lambda: brian_prefs.weave_compiler)
brian_prefs.define('fake_weave_compiler', 'gcc',
                   'The distutils compiler specifier for weave compilation.')
brian_prefs.fake_weave_compiler
brian_prefs.fake_weave_compiler = 'msvc'
brian_prefs.fake_weave_compiler
assert_raises(TypeError, lambda: setattr(brian_prefs, 'fake_weave_compiler', 5))
brian_prefs.define('fake_weave_compiler_options', ['-O3', '-ffast-math'],
                   'The extra compiler options for weave compilation.')
brian_prefs.documentation

if __name__=='__main__':
    try:
        print brian_prefs.weave_compiler
    except KeyError:
        print 'Key error raised correctly.'
    brian_prefs.define('weave_compiler', 'gcc',
                       'The distutils compiler specifier for weave compilation.')
    print 'Compiler:', brian_prefs.weave_compiler
    brian_prefs.weave_compiler = 'msvc'
    print 'Compiler:', brian_prefs.weave_compiler
    try:
        brian_prefs.weave_compiler = 5
    except TypeError as e:
        print 'Type error correct,', str(e)
    brian_prefs.define('weave_compiler_options', ['-O3', '-ffast-math'],
                       'The extra compiler options for weave compilation.')
    print 'Documentation'
    print brian_prefs.documentation
