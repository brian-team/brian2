'''
Tool to set the current version of Brian in the various places that have it,
i.e.:

* Global __version__ in __init__.py
* setup.py version
* docs version
* README.txt version
'''

import os, sys, re, datetime


def setversion(version):
    docs_release = version
    major = docs_release[:docs_release.find('.')]
    minor = docs_release[docs_release.find('.') + 1:docs_release.find('.', docs_release.find('.') + 1)]
    docs_version = major + '.' + minor
    pathname = os.path.abspath(os.path.dirname(__file__))
    os.chdir(pathname)
    os.chdir('../../../')
    # update setup.py
    setup_py = open('setup.py', 'r').read()
    setup_py = re.sub("version\s*=\s*'.*?'", "version='" + version + "'", setup_py)
    open('setup.py', 'w').write(setup_py)
    # update __init__.py
    init_py = open('brian2/__init__.py', 'r').read()
    init_py = re.sub("__version__\s*=\s*'.*?'", "__version__ = '" + version + "'", init_py)
    open('brian2/__init__.py', 'w').write(init_py)
    # update sphinx docs
    conf_py = open('docs_sphinx/conf.py', 'r').read()
    conf_py = re.sub("version\s*=\s*'.*?'", "version = '" + docs_version + "'", conf_py)
    conf_py = re.sub("release\s*=\s*'.*?'", "release = '" + docs_release + "'", conf_py)
    open('docs_sphinx/conf.py', 'w').write(conf_py)
    # update conda recipe
    meta_yaml = open('dev/conda-recipe/meta.yaml', 'r').read()
    meta_yaml = re.sub('version\s*:\s*".*?"', 'version: "' + version + '"', meta_yaml)
    open('dev/conda-recipe/meta.yaml', 'w').write(meta_yaml)


def setreleasedate():
    releasedate = str(datetime.date.today())
    pathname = os.path.abspath(os.path.dirname(__file__))
    os.chdir(pathname)
    os.chdir('../../../')
    # update __init__.py
    init_py = open('brian2/__init__.py', 'r').read()
    init_py = re.sub("__release_date__\s*=\s*'.*?'", "__release_date__ = '" + releasedate + "'", init_py)
    open('brian2/__init__.py', 'w').write(init_py)
