'''
Generate the brian2/default_preferences file automatically from the default
values registered when you import Brian.
'''

import os
import brian2

base, _ = os.path.split(brian2.__file__)
fname = os.path.join(base, 'default_preferences')

open(fname, 'w').write(brian2.prefs.as_file)
