import os
import shutil
import glob
import codecs

from nbformat.v4 import reads
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.exporters.notebook import NotebookExporter
from nbconvert.exporters.rst import RSTExporter

from brian2.utils.stringtools import deindent, indent


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
    with open(os.path.join(src_dir, fname), 'r') as f:
        notebook = reads(f.read())

    # The first line of the tutorial file should give the title
    title = notebook.cells[0]['source'].split('\n')[0].strip('# ')
    tutorials.append((basename, title))

    # Execute the notebook
    preprocessor = ExecutePreprocessor()
    preprocessor.allow_errors = True
    notebook, _ = preprocessor.preprocess(notebook,
                                          {'metadata': {'path': src_dir}})

    print 'Saving notebook and converting to RST'
    exporter = NotebookExporter()
    output, _ = exporter.from_notebook_node(notebook)
    with codecs.open(output_ipynb_fname, 'w', encoding='utf-8') as f:
        f.write(output)

    # Insert a note about ipython notebooks with a download link
    note = deindent(u'''
    .. only:: html

        .. |launchbinder| image:: http://mybinder.org/badge.svg
        .. _launchbinder: http://mybinder.org:/repo/brian-team/brian2-binder/notebooks/tutorials/{tutorial}.ipynb
    
        .. note::
           This tutorial is a static non-editable version. You can launch an
           interactive, editable version without installing any local files
           using the Binder service (although note that at some times this
           may be slow or fail to open): |launchbinder|_
    
           Alternatively, you can download a copy of the notebook file
           to use locally: :download:`{tutorial}.ipynb`
    
           See the :doc:`tutorial overview page <index>` for more details.

    '''.format(tutorial=basename))
    notebook.cells.insert(1, {
        u'cell_type': u'raw',
        u'metadata': {},
        u'source': note
    })

    exporter = RSTExporter()
    output, resources = exporter.from_notebook_node(notebook,
                                                    resources={'unique_key': basename+'_image'})
    with codecs.open(output_rst_fname, 'w', encoding='utf-8') as f:
        f.write(output)

    for image_name, image_data in resources['outputs'].iteritems():
        with open(os.path.join(target_dir, image_name), 'wb') as f:
            f.write(image_data)

print 'Generating index.rst'

text = '''
..
    This is a generated file, do not edit directly.
    (See dev/tools/docs/build_tutorials.py)

Tutorials
=========

The tutorial consists of a series of `Jupyter Notebooks`_ [#]_.

.. only:: html

    You can quickly view these using the first links below. To use them interactively - allowing you
    to edit and run the code - there are two options. The easiest option is to click
    on the "Launch Binder" link, which will open up an interactive version in the
    browser without having to install Brian locally. This uses the
    Binder service provided by the
    `Freeman lab <https://www.janelia.org/lab/freeman-lab>`_. Occasionally, this
    service will be down or running slowly. The other option is to download the
    notebook file and run it locally, which requires you to have Brian installed.

For more information about how to use Jupyter Notebooks, see the
`Jupyter Notebook documentation`_.

.. toctree::
   :maxdepth: 1
   :titlesonly:

'''
for tutorial, _ in tutorials:
    text += '   ' + tutorial + '\n'
text += '''
.. only:: html

    Interactive notebooks and files
    -------------------------------
'''
for tutorial, _ in tutorials:
    text += indent(deindent('''
    .. |launchbinder{tutid}| image:: http://mybinder.org/badge.svg
    .. _launchbinder{tutid}: http://mybinder.org:/repo/brian-team/brian2-binder/notebooks/tutorials/{tutorial}.ipynb
    '''.format(tutorial=tutorial, tutid=tutorial.replace('-', ''))))

text += '\n'
for tutorial, title in tutorials:
    text += '    * |launchbinder{tutid}|_ :download:`{title} <{tutorial}.ipynb>`\n'.format(title=title,
                                                tutorial=tutorial, tutid=tutorial.replace('-', ''))
text += '''

.. _`Jupyter Notebooks`: http://jupyter-notebook-beginner-guide.readthedocs.org/en/latest/what_is_jupyter.html
.. _`Jupyter`: http://jupyter.org/
.. _`Jupyter Notebook documentation`: http://jupyter.readthedocs.org/

.. [#] Formerly known as "IPython Notebooks".
'''
with open(os.path.join(target_dir, 'index.rst'), 'w') as f:
    f.write(text)
