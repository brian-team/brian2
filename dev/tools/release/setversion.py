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
    version_parts = version.split('.')
    assert len(version_parts) >= 2
    docs_release = version
    docs_version = version_parts[0] + '.' + version_parts[1]
    pathname = os.path.abspath(os.path.dirname(__file__))
    os.chdir(pathname)
    os.chdir('../../../')
    # update setup.py
    with open('setup.py', 'r') as f:
        setup_py = f.read()
    setup_py = re.sub("version\s*=\s*'.*?'", "version='" + version + "'", setup_py)
    with open('setup.py', 'w') as f:
        f.write(setup_py)
    # update __init__.py
    with open('brian2/__init__.py', 'r') as f:
        init_py = f.read()
    init_py = re.sub("__version__\s*=\s*'.*?'", "__version__ = '" + version + "'", init_py)
    with open('brian2/__init__.py', 'w') as f:
        f.write(init_py)
    # update sphinx docs
    with open('docs_sphinx/conf.py', 'r') as f:
        conf_py = f.read()
    conf_py = re.sub("version\s*=\s*'.*?'", "version = '" + docs_version + "'", conf_py)
    conf_py = re.sub("release\s*=\s*'.*?'", "release = '" + docs_release + "'", conf_py)
    with open('docs_sphinx/conf.py', 'w') as f:
        f.write(conf_py)
    # update conda recipe
    with open('dev/conda-recipe/meta.yaml', 'r') as f:
        meta_yaml = f.read()
    meta_yaml = re.sub('version\s*:\s*".*?"', 'version: "' + version + '"', meta_yaml)
    with open('dev/conda-recipe/meta.yaml', 'w') as f:
        f.write(meta_yaml)


def setreleasedate():
    releasedate = str(datetime.date.today())
    pathname = os.path.abspath(os.path.dirname(__file__))
    os.chdir(pathname)
    os.chdir('../../../')
    # update __init__.py
    with open('brian2/__init__.py', 'r') as f:
        init_py = f.read()
    init_py = re.sub("__release_date__\s*=\s*'.*?'", "__release_date__ = '" + releasedate + "'", init_py)
    with open('brian2/__init__.py', 'w') as f:
        f.write(init_py)


if __name__ == '__main__':
    version = raw_input('New version: ')
    setversion(version)
