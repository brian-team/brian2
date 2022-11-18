import os
import fnmatch
import shutil
from collections import defaultdict
import glob
import codecs


class GlobDirectoryWalker:
    # a forward iterator that traverses a directory tree

    def __init__(self, directory, pattern="*"):
        self.stack = [directory]
        self.pattern = pattern
        self.files = []
        self.index = 0

    def __getitem__(self, index):
        while True:
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

    examplesfnames = [fname for fname in GlobDirectoryWalker(rootpath, "*.py")]
    additional_files = [
        fname
        for fname in GlobDirectoryWalker(rootpath, "*.[!py]*")
        if not os.path.basename(fname) == ".gitignore"
    ]

    print("Documenting %d examples" % len(examplesfnames))

    examplespaths = []
    examplesbasenames = []
    relativepaths = []
    outnames = []
    for f in examplesfnames:
        path, file = os.path.split(f)
        relpath = os.path.relpath(path, rootpath)
        if relpath == ".":
            relpath = ""
        path = os.path.normpath(path)
        filebase, ext = os.path.splitext(file)
        exname = filebase
        if relpath:
            exname = relpath.replace("/", ".").replace("\\", ".") + "." + exname
        examplespaths.append(path)
        examplesbasenames.append(filebase)
        relativepaths.append(relpath)
        outnames.append(exname)
    # We assume all files are encoded as UTF-8
    examplescode = []
    for fname in examplesfnames:
        with codecs.open(fname, "rU", encoding="utf-8") as f:
            examplescode.append(f.read())
    examplesdocs = []
    examplesafterdoccode = []
    examplesdocumentablenames = []
    for code in examplescode:
        codesplit = code.split("\n")
        comment_lines = 0
        for line in codesplit:
            if line.startswith("#") or len(line) == 0:
                comment_lines += 1
            else:
                break
        codesplit = codesplit[comment_lines:]
        readingdoc = False
        doc = []
        afterdoccode = ""
        for i in range(len(codesplit)):
            stripped = codesplit[i].strip()
            if stripped[:3] == '"""' or stripped[:3] == "'''":
                if not readingdoc:
                    readingdoc = True
                else:
                    afterdoccode = "\n".join(codesplit[i + 1 :])
                    break
            elif readingdoc:
                doc.append(codesplit[i])
            else:  # No doc
                afterdoccode = "\n".join(codesplit[i:])
                break
        examplesdocs.append("\n".join(doc))
        examplesafterdoccode.append(afterdoccode)

    categories = defaultdict(list)
    examples = zip(
        examplesfnames,
        examplespaths,
        examplesbasenames,
        examplescode,
        examplesdocs,
        examplesafterdoccode,
        relativepaths,
        outnames,
    )
    # Get the path relative to the examples director (not relative to the
    # directory where this file is installed
    if "BRIAN2_DOCS_EXAMPLE_DIR" in os.environ:
        rootdir = os.environ["BRIAN2_DOCS_EXAMPLE_DIR"]
    else:
        rootdir, _ = os.path.split(__file__)
        rootdir = os.path.normpath(os.path.join(rootdir, "../../examples"))
    eximgpath = os.path.abspath(
        os.path.join(rootdir, "../docs_sphinx/resources/examples_images")
    )
    print("Searching for example images in directory", eximgpath)
    for fname, path, basename, code, docs, afterdoccode, relpath, exname in examples:
        categories[relpath].append((exname, basename))
        title = "Example: " + basename
        output = ".. currentmodule:: brian2\n\n"
        output += ".. " + basename + ":\n\n"
        output += title + "\n" + "=" * len(title) + "\n\n"
        note = f"""
        .. only:: html

            .. |launchbinder| image:: http://mybinder.org/badge.svg
            .. _launchbinder: https://mybinder.org/v2/gh/brian-team/brian2-binder/master?filepath=examples/{exname.replace('.', '/')}.ipynb

            .. note::
               You can launch an interactive, editable version of this
               example without installing any local files
               using the Binder service (although note that at some times this
               may be slow or fail to open): |launchbinder|_

        """
        output += note + "\n\n"
        output += docs + "\n\n::\n\n"
        output += "\n".join(["    " + line for line in afterdoccode.split("\n")])
        output += "\n\n"

        eximgpattern = os.path.join(eximgpath, "%s.*.png" % exname)
        images = glob.glob(eximgpattern)
        for image in sorted(images):
            _, image = os.path.split(image)
            print("Found example image file", image)
            output += ".. image:: ../resources/examples_images/%s\n\n" % image

        with codecs.open(os.path.join(destdir, exname + ".rst"), "w", "utf-8") as f:
            f.write(output)

    category_additional_files = defaultdict(list)
    for fname in additional_files:
        path, file = os.path.split(fname)
        relpath = os.path.relpath(path, rootpath)
        if relpath == ".":
            relpath = ""
        full_name = relpath.replace("/", ".").replace("\\", ".") + "." + file + ".rst"
        category_additional_files[relpath].append((file, full_name))
        with codecs.open(fname, "rU", encoding="utf-8") as f:
            print(fname)
            content = f.read()
        output = file + "\n" + "=" * len(file) + "\n\n"
        output += ".. code:: none\n\n"
        content_lines = ["\t" + l for l in content.split("\n")]
        output += "\n".join(content_lines)
        output += "\n\n"
        with codecs.open(os.path.join(destdir, full_name), "w", "utf-8") as f:
            f.write(output)

    mainpage_text = "Examples\n"
    mainpage_text += "========\n\n"

    def insert_category(category, mainpage_text):
        if category:
            label = category.lower().replace(" ", "-").replace("/", ".")
            mainpage_text += f"\n.. _{label}:\n\n"
            mainpage_text += "\n" + category + "\n" + "-" * len(category) + "\n\n"
        mainpage_text += ".. toctree::\n"
        mainpage_text += "   :maxdepth: 1\n\n"
        curpath = ""
        for exname, basename in sorted(categories[category]):
            mainpage_text += f"   {basename} <{exname}>\n"
        for fname, full_name in sorted(category_additional_files[category]):
            mainpage_text += f"   {fname} <{full_name}>\n"
        return mainpage_text

    mainpage_text = insert_category("", mainpage_text)
    for category in sorted(categories.keys()):
        if category:
            mainpage_text = insert_category(category, mainpage_text)

    with open(os.path.join(destdir, "index.rst"), "w") as f:
        f.write(mainpage_text)


if __name__ == "__main__":
    main("../../examples", "../../docs_sphinx/examples")
