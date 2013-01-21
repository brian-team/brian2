import os
import shutil

os.chdir('../../../docs_sphinx')
if os.path.exists('../docs'):
    shutil.rmtree('../docs')
os.system('sphinx-build -b html . ../docs')
