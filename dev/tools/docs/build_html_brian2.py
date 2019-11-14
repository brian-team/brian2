import os
import shutil
import sphinx
if not sphinx.version_info >= (1, 8):
    raise ImportError('Need sphinx version 1.8')
from sphinx.cmd.build import main as sphinx_main
import sys

os.chdir('../../../docs_sphinx')
if os.path.exists('../docs'):
    shutil.rmtree('../docs')
sys.exit(sphinx_main(['-b', 'html', '.', '../docs']))
