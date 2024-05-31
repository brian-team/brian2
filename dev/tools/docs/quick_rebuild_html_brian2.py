import os
import shutil
import sphinx
if not sphinx.version_info >= (1, 8):
    raise ImportError('Need sphinx version 1.8')
from sphinx.cmd.build import main as sphinx_main
import sys

os.environ['BRIAN2_DOCS_QUICK_REBUILD'] = '1'
# Some code (e.g. the definition of preferences) might need to know that Brian
# is used to build the documentation. The READTHEDOCS variable is set
# on the readthedocs.io server automatically, so we reuse it here to signal
# a documentation build
os.environ['READTHEDOCS'] = 'True'

os.chdir(os.path.join(os.path.dirname(__file__), '../../../docs_sphinx'))
if os.path.exists('../docs'):
    shutil.rmtree('../docs')
sys.exit(sphinx_main(['-b', 'html', '.', '../docs']))
