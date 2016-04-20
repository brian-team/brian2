import os
import shutil
import glob
import codecs

from nbformat.v4 import reads
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.exporters.notebook import NotebookExporter
from nbconvert.exporters.rst import RSTExporter


src_dir = os.path.abspath('../../../tutorials')
target_dir = os.path.abspath('../../../docs_sphinx/resources/tutorials')

# Start from scratch to avoid left-over files due to renamed tutorials, changed
# cell numbers, etc.
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)

tutorials = []
for fname in sorted(glob.glob1(src_dir, '*.ipynb')):
    basename = fname[:-6]
    output_ipynb_fname = os.path.join(target_dir, fname)
    output_rst_fname = os.path.join(target_dir, basename + '.rst')

    print 'Running', fname
    notebook = reads(open(os.path.join(src_dir, fname), 'r').read())

    # The first line of the tutorial file should give the title
    title = notebook.cells[0]['source'].split('\n')[0].strip('# ')
    tutorials.append((basename, title))

    # Execute the notebook
    preprocessor = ExecutePreprocessor()
    preprocessor.allow_errors = True
    notebook, _ = preprocessor.preprocess(notebook, {})

    print 'Saving notebook and converting to RST'
    exporter = NotebookExporter()
    output, _ = exporter.from_notebook_node(notebook)
    codecs.open(output_ipynb_fname, 'w', encoding='utf-8').write(output)

    # Insert a note about ipython notebooks with a download link
    note = u'''
    .. note::
       This tutorial is written as an interactive notebook that should be run
       on your own computer. See the :doc:`tutorial overview page <index>` for
       more details.

       Download link for this tutorial: :download:`{tutorial}.ipynb`.
    '''.format(tutorial=basename)
    notebook.cells.insert(1, {
        u'cell_type': u'raw',
        u'metadata': {},
        u'source': note
    })

    exporter = RSTExporter()
    output, resources = exporter.from_notebook_node(notebook,
                                                    resources={'unique_key': basename+'_image'})
    codecs.open(output_rst_fname, 'w', encoding='utf-8').write(output)

    for image_name, image_data in resources['outputs'].iteritems():
        open(os.path.join(target_dir, image_name), 'wb').write(image_data)

print 'Generating index.rst'

text = '''
..
    This is a generated file, do not edit directly.
    (See dev/tools/docs/build_tutorials.py)

Tutorials
=========

The tutorial consists of a series of `Jupyter Notebooks`_ [#]_. If you run such
a notebook on your own computer, you can interactively change the code in the
tutorial and experiment with it -- this is the recommended way to get started
with Brian. The first link for each tutorial below leads to a non-interactive
version of the notebook; use the links under "Notebook files" to get a file that
you can run on your computer. You can also copy such a link and paste it at
http://nbviewer.jupyter.org -- this will get you a nicer (but still
non-interactive) rendering than the one you see in our documentation.

For more information about how to use Jupyter Notebooks, see the
`Jupyter Notebook documentation`_.

.. toctree::
   :maxdepth: 1
   :titlesonly:

'''
for tutorial, _ in tutorials:
    text += '   ' + tutorial + '\n'
text += '''

Notebook files
--------------
'''
for tutorial, title in tutorials:
    text += '* :download:`{title} <{tutorial}.ipynb>`\n'.format(title=title,
                                                                tutorial=tutorial)
text += '''

.. _`Jupyter Notebooks`: http://jupyter-notebook-beginner-guide.readthedocs.org/en/latest/what_is_jupyter.html
.. _`Jupyter`: http://jupyter.org/
.. _`Jupyter Notebook documentation`: http://jupyter.readthedocs.org/

.. [#] Formerly known as "IPython Notebooks".
'''
open(os.path.join(target_dir, 'index.rst'), 'w').write(text)
