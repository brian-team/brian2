import os
import fnmatch
import shutil
import unicodedata
from collections import defaultdict
import glob


class GlobDirectoryWalker:
    # a forward iterator that traverses a directory tree

    def __init__(self, directory, pattern="*"):
        self.stack = [directory]
        self.pattern = pattern
        self.files = []
        self.index = 0

    def __getitem__(self, index):
        while 1:
            try:
                file = self.files[self.index]
                self.index = self.index + 1
            except IndexError:
                # pop next directory from stack
                self.directory = self.stack.pop()
                if os.path.isdir(self.directory):
                    self.files = os.listdir(self.directory)
                else:
                    self.files = []
                self.index = 0
            else:
                # got a filename
                fullname = os.path.join(self.directory, file)
                if os.path.isdir(fullname) and not os.path.islink(fullname):
                    self.stack.append(fullname)
                if fnmatch.fnmatch(file, self.pattern):
                    return fullname


def main(rootpath, destdir):
    if not os.path.exists(destdir):
        shutil.os.makedirs(destdir)

    examplesfnames = [fname for fname in GlobDirectoryWalker(rootpath, '*.py')]
    print 'Documenting %d examples' % len(examplesfnames)
    examplespaths = []
    examplesbasenames = []
    relativepaths = []
    outnames = []
    for f in examplesfnames:
        path, file = os.path.split(f)
        relpath = os.path.relpath(path, rootpath)
        if relpath=='.':
            relpath = ''
        path = os.path.normpath(path)
        filebase, ext = os.path.splitext(file)
        exname = filebase
        if relpath:
            exname = relpath.replace('/', '.').replace('\\', '.')+'.'+exname
        examplespaths.append(path)
        examplesbasenames.append(filebase)
        relativepaths.append(relpath)
        outnames.append(exname)
    examplescode = [open(fname, 'rU').read() for fname in examplesfnames]
    examplesdocs = []
    examplesafterdoccode = []
    examplesdocumentablenames = []
    for code in examplescode:
        codesplit = code.split('\n')
        if codesplit[0].startswith('#'):
            codesplit = codesplit[1:]
        readingdoc = False
        doc = []
        afterdoccode = ''
        for i in range(len(codesplit)):
            stripped = codesplit[i].strip()
            if stripped[:3] == '"""' or stripped[:3] == "'''":
                if not readingdoc:
                    readingdoc = True
                else:
                    afterdoccode = '\n'.join(codesplit[i + 1:])
                    break
            elif readingdoc:
                doc.append(codesplit[i])
            else: # No doc
                afterdoccode = '\n'.join(codesplit[i:])
                break
        doc = '\n'.join(doc)
        # next line replaces unicode characters like e-acute with standard ascii representation
        examplesdocs.append(unicodedata.normalize('NFKD', unicode(doc, 'latin-1')).encode('ascii', 'ignore'))
        examplesafterdoccode.append(afterdoccode)
        
    categories = defaultdict(list)    
    examples = zip(examplesfnames, examplespaths, examplesbasenames,
                   examplescode, examplesdocs, examplesafterdoccode,
                   relativepaths, outnames)
    # Get the path relative to the examples director (not relative to the
    # directory where this file is installed
    if 'BRIAN2_DOCS_EXAMPLE_DIR' in os.environ:
        rootdir = os.environ['BRIAN2_DOCS_EXAMPLE_DIR']
    else:
        rootdir, _ = os.path.split(__file__)
        rootdir = os.path.normpath(os.path.join(rootdir, '../../examples'))
    eximgpath = os.path.abspath(os.path.join(rootdir, '../docs_sphinx/resources/examples_images'))
    print 'Searching for example images in directory', eximgpath
    for fname, path, basename, code, docs, afterdoccode, relpath, exname in examples:
        categories[relpath].append((exname, basename))
        title = 'Example: ' + basename
        output = '.. currentmodule:: brian2\n\n'
        output += '.. ' + basename + ':\n\n'
        output += title + '\n' + '=' * len(title) + '\n\n'
        output += docs + '\n\n::\n\n'
        output += '\n'.join(['    ' + line for line in afterdoccode.split('\n')])
        output += '\n\n'

        eximgpattern = os.path.join(eximgpath, '%s.*.png' % exname)
        images = glob.glob(eximgpattern)
        for image in sorted(images):
            _, image = os.path.split(image)
            print 'Found example image file', image
            output += '.. image:: ../resources/examples_images/%s\n\n' % image
    
        open(os.path.join(destdir, exname + '.rst'), 'w').write(output)
    
    mainpage_text =  'Examples\n'
    mainpage_text += '========\n\n'
    
    def insert_category(category, mainpage_text):
        if category:
            mainpage_text += '\n'+category+'\n'+'-'*len(category)+'\n\n'
        mainpage_text += '.. toctree::\n'
        mainpage_text += '   :maxdepth: 1\n\n'
        curpath = ''
        for exname, basename in sorted(categories[category]):
            mainpage_text += '   %s <%s>\n' % (basename, exname)
        return mainpage_text
            
    mainpage_text = insert_category('', mainpage_text)
    for category in sorted(categories.keys()):
        if category:
            mainpage_text = insert_category(category, mainpage_text)
            
    open(os.path.join(destdir, 'index.rst'), 'w').write(mainpage_text)
    

if __name__=='__main__':
    main('../../examples', '../../docs_sphinx/examples')
    