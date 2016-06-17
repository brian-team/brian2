import os

import sphinx

os.chdir('../../../docs_sphinx')
sphinx.main(['sphinx-build', '-b', 'doctest', '.', '../docs', '-D',
             'exclude_patterns=reference'])
