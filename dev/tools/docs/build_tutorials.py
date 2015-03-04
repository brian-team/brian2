import os
import shutil
import glob
import codecs

from IPython.nbformat.v4 import reads
from IPython.nbconvert.preprocessors import ExecutePreprocessor
from IPython.nbconvert.exporters.rst import RSTExporter


src_dir = os.path.abspath('../../../tutorials')
target_dir = os.path.abspath('../../../docs_sphinx/tutorials')

if not os.path.exists(target_dir):
    os.mkdir(target_dir)


def print_dot(cell_no):
    # Simple callback function to give some indication of progress
    print '.',

tutorials = []
for fname in sorted(glob.glob1(src_dir, '*.ipynb')):
    basename = fname[:-6]
    image_dir = basename+'_images'
    output_fname = os.path.join(target_dir, basename + '.rst')
    tutorials.append(basename)
    print 'Running', fname
    notebook = reads(open(os.path.join(src_dir, fname), 'r').read())
    preprocessor = ExecutePreprocessor()
    notebook, _ = preprocessor.preprocess(notebook, {})
    print 'Converting to RST'
    exporter = RSTExporter()
    output, resources = exporter.from_notebook_node(notebook,
                                                    resources={'output_files_dir': image_dir})
    codecs.open(output_fname, 'w', encoding='utf-8').write(output)

    full_image_dir = os.path.join(target_dir, image_dir)
    if os.path.exists(full_image_dir):
        shutil.rmtree(full_image_dir)
    os.mkdir(full_image_dir)
    for image_name, image_data in resources['outputs'].iteritems():
        open(os.path.join(target_dir, image_name), 'wb').write(image_data)

print 'Generating index file'
text = '''
Tutorial
========

.. toctree::
   :maxdepth: 1
   :titlesonly:

'''
for tutorial in tutorials:
    text += '   ' + tutorial + '\n'
text += '\n'
open(os.path.join(target_dir, 'index.rst'), 'w').write(text)
