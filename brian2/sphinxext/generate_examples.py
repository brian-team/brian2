import os
import fnmatch
import shutil
import unicodedata


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
                self.files = os.listdir(self.directory)
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
    for f in examplesfnames:
        path, file = os.path.split(f)
        path = os.path.normpath(path)
        filebase, ext = os.path.splitext(file)
        examplespaths.append(path)
        examplesbasenames.append(filebase)
    examplescode = [open(fname, 'rU').read() for fname in examplesfnames]
    examplesdocs = []
    examplesafterdoccode = []
    examplesdocumentablenames = []
    for code in examplescode:
        codesplit = code.split('\n')
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
    
    examples = zip(examplesfnames, examplespaths, examplesbasenames, examplescode, examplesdocs, examplesafterdoccode)
    for fname, path, basename, code, docs, afterdoccode in examples:
        title = 'Example: ' + basename
        output = '.. currentmodule:: brian2\n\n'
        output += '.. ' + basename + ':\n\n'
        output += title + '\n' + '=' * len(title) + '\n\n'
        output += docs + '\n\n::\n\n'
        output += '\n'.join(['    ' + line for line in afterdoccode.split('\n')])
        output += '\n\n'
    
        open(os.path.join(destdir, basename + '.rst'), 'w').write(output)
    
    mainpage_text =  'Examples\n'
    mainpage_text += '========\n\n'
    mainpage_text += '.. toctree::\n'
    mainpage_text += '   :maxdepth: 1\n\n'
    curpath = ''
    for fname, path, basename, code, docs, afterdoccode in examples:
        mainpage_text += '   ' + basename + '\n'
    open(os.path.join(destdir, 'index.rst'), 'w').write(mainpage_text)
    
