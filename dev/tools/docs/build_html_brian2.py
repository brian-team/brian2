import os
import shutil
import sphinx
import sys

os.chdir('../../../docs_sphinx')
if os.path.exists('../docs'):
    shutil.rmtree('../docs')
sys.exit(sphinx.main(['sphinx-build', '-b', 'html', '.', '../docs']))
