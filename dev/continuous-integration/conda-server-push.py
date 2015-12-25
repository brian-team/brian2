import sys
import os
import glob
import time

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

# Uploading sometimes fails due to server or network errors -- we try it five
# times before giving up
attempts = 5
while attempt in range(attempts):
    return_value = main(args=options)
    if return_value == 0:
        # all good
        break
    else:
        if attempt < attempts - 1:
            print('Something did not work, trying again in 10 seconds.')
            time.sleep(10)
        else:
            print('Giving up...')
            sys.exit(return_value)
