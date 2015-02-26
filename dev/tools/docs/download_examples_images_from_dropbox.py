'''
Script to download and extract saved examples images from a shared Dropbox
file location. This is used for the documentation system. Images generated
on one developer's computer are copied to a dropbox file (by the script
copy_examples_images_to_dropbox.py) and then readthedocs script can just
download the images (which can take a couple of hours to generate so that
wouldn't work directly on readthedocs).
'''

import os, zipfile, urllib2

zipfile_url = 'https://www.dropbox.com/s/uqb44b9t1gpvy61/doc_example_images.zip?dl=1'

def download_examples_images_from_dropbox():
    outputdir = '../../../docs_sphinx/examples_images'
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    outfname = os.path.join(outputdir, 'doc_example_images.zip')
    f = urllib2.urlopen(zipfile_url)
    open(outfname, 'wb').write(f.read())
    z = zipfile.ZipFile(outfname, 'r')
    z.extractall(outputdir)
    
if __name__=='__main__':
    download_examples_images_from_dropbox()
    