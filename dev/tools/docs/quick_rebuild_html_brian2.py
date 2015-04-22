import os
import shutil
import sphinx
import sys

os.environ['BRIAN2_DOCS_QUICK_REBUILD'] = '1'
os.chdir('../../../docs_sphinx')
sys.exit(sphinx.main(['sphinx-build', '-b', 'html', '.', '../docs']))
