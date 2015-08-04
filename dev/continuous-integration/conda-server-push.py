import sys
import os
import glob

from binstar_client.scripts.cli import main

token = os.environ['BINSTAR_TOKEN']
options = ['-t', token, 'upload',
           '-u', 'brian-team']
filename = glob.glob('*.tar.bz2')
assert len(filename) == 1, 'Expected to find one .tar.bz2 file, found %d' % len(filename)
release = '+git' not in filename[0]
if not release:
    options.extend(['--channel', 'dev', '--force'])

options.extend(filename)

sys.exit(main(args=options))
