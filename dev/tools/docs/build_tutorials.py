import os
import shutil
import glob
import codecs

from IPython.nbformat.v4 import reads
from IPython.nbconvert.preprocessors import ExecutePreprocessor
from IPython.nbconvert.exporters.notebook import NotebookExporter
from IPython.nbconvert.exporters.rst import RSTExporter


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

The tutorial consists of a series of `IPython notebooks`_ [#]_. If you run such
a notebook on your own computer, you can interactively change the code in the
tutorial and experiment with it -- this is the recommended way to get started
with Brian. The first link for each tutorial below leads to a non-interactive
version of the notebook; use the links under "Notebook files" to get a file that
you can run on your computer. You can also copy such a link and paste it at
http://nbviewer.ipython.org -- this will get you a nicer (but still
non-interactive) rendering than the one you see in our documentation.

For more information about how to use IPython notebooks, see the
`IPython notebook documentation`_.

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

.. _`IPython notebooks`: http://ipython.org/notebook.html
.. _`Jupyter`: http://jupyter.org/
.. _`IPython notebook documentation`: http://ipython.org/ipython-doc/stable/notebook/index.html

.. [#] The project has been partly renamed to `Jupyter`_ recently
'''
open(os.path.join(target_dir, 'index.rst'), 'w').write(text)
