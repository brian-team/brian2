import brian2
import sphinx
import doc2dash
import sys
import os

# - Build documentation with Sphinx
os.chdir('../../../docs_sphinx')
sphinx.main(['sphinx-build', '-b', 'html', '-d', '_build/doctrees', '.', '_build/html'])

# - Run doc2dash on the built documentation
os.system('doc2dash _build/html/ -n Brian2 -I index.html -d ..')

