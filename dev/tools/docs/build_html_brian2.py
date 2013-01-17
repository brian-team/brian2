import os
import shutil

os.chdir('../../../docs_sphinx')
shutil.rmtree('../docs')
os.system('sphinx-build -b html . ../docs')
