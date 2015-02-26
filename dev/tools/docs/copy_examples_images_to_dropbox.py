'''
This script copies the examples_images to a zipfile in a dropbox location
so that it can be read by download_examples_images_from_dropbox.py. This
should be run after a successful run of run_examples.py.
'''

import os, sys, zipfile, glob

__all__ = ['copy_examples_images_to_dropbox']

# add your dropbox location here
possible_dropbox_directories = [
    '~/Dropbox/Projects/Shared Brian/Documentation example images',
    ]

def copy_examples_images_to_dropbox():
    
    zipname = None
    for dirname in possible_dropbox_directories:
        dirname = os.path.expanduser(dirname)
        if os.path.exists(dirname):
            zipname = os.path.join(dirname, 'doc_example_images.zip')
    
    if zipname is None:
        print 'Cannot find Dropbox directory'
        return
    
    imgs = glob.glob('../../../docs_sphinx/examples_images/*.png')
    
    with zipfile.ZipFile(zipname, 'w') as z:
        for f in imgs:
            base, fname = os.path.split(f)
            z.write(f, fname)


if __name__=='__main__':
    copy_examples_images_to_dropbox()
    