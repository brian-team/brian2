import os
import shutil
import generate_reference
# first generate the reference documentation
abs_root = os.path.abspath('../../../brian2')
target_dir = '../../../docs_sphinx/reference'
shutil.rmtree('../../../docs_sphinx/reference')
os.makedirs(target_dir)
generate_reference.main(abs_root, ['tests'], target_dir)

os.chdir('../../../docs_sphinx')
shutil.rmtree('../docs')
os.system('sphinx-build -b html . ../docs')
