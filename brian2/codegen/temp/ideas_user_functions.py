'''
The following shows a scheme we could use to allow user-defined functions in
C++ using weave. It relies on the user providing the following elements:

- definition of _func_f which takes all arguments needed, i.e. any values from
  the global namespace which will not be accessible to the function definition
- #define mapping from f(t) to _func_f form, so that in the code f(t) can be
  used directly
- values to insert into namespace (_len_f, _values_f)

So a scheme that can handle this would be:

User provides:
- Function definition code that is inserted into support_code, the user code
  is provided the name of the variable which they should use in defining their
  names, according to the scheme _func_{name}
- #define code which should be of the form #define {name}(args) = _func_{name}(args, extraargs)
- Code to update the namespace at compilation time (and maybe each time it is
  called?)
  
This suggests a class like:

class UserFunction(object):
    def code(language, var):
        """
        Returns a dict of (slot, section) values, where slot is a
        language-specific slot for where to include the string section. The
        input arguments are the language object, and the variable name.
        """
        raise NotImplementedError
        
    def on_compile(language, var, namespace):
        """
        What to do at compile time, i.e. insert values into a namespace.
        """
        raise NotImplementedError
        
This should work in Python and for GPU as well, although for GPU we need to
think carefully about namespace management of course.

Because of the use of support_code as well as the main code, this suggests
that the code returned should not necessarily just be a single string but also
possibly a dict of (slot, section) items, or something like that.
'''

from numpy import *
from scipy import weave

support_code = '''
inline double _func_f(const double t, const double dt, const int _len_f, const double* _values_f)
{
    int i = (int)((t+0.5)/dt); // rounds to nearest int for positive values
    if(i<0) i = 0;
    if(i>=_len_f) i = _len_f-1;
    return _values_f[i];
}
'''

code = '''
#define f(t) _func_f(t, dt, _len_f, _values_f)
I[0] = f(t);
'''

I = zeros(1)
t = 0.3
dt = 0.1
_values_f = arange(int(ceil(t/dt))+1)*dt
_len_f = len(_values_f)

weave.inline(code, ['I', 't', 'dt', '_values_f', '_len_f'],
             support_code=support_code)

print I[0]
