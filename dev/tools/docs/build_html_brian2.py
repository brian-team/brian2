import os
os.chdir('../../../dev/brian2/docs_sphinx')
os.system('sphinx-build -b html . ../docs')
