import os
os.chdir('../../../docs_sphinx')
os.system('sphinx-build -b html . ../docs')
