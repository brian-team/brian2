import os
import shutil
import sphinx
import sys
from download_examples_images_from_dropbox import download_examples_images_from_dropbox 

download_examples_images_from_dropbox()
os.chdir('../../../docs_sphinx')
if os.path.exists('../docs'):
    shutil.rmtree('../docs')
sys.exit(sphinx.main(['sphinx-build', '-b', 'html', '.', '../docs']))
