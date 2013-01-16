import os
import shutil
import generate_references
# first generate the reference documentation
abs_root = os.path.abspath('../../../brian2')
generate_references.main(abs_root, ['tests'], '../../../docs_sphinx/reference')

os.chdir('../../../docs_sphinx')
shutil.rmtree('../docs')
os.system('sphinx-build -b html . ../docs')
