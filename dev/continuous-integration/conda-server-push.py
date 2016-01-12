import sys
import os
import glob
import time

from binstar_client.scripts.cli import main
from binstar_client.errors import BinstarError

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
for attempt in range(attempts):
    try:
        main(args=options)
        break  # all good
    except BinstarError as ex:
        print('Something did not work (%s).' % str(ex))
        if attempt < attempts - 1:
            print('Trying again in 10 seconds...')
            time.sleep(10)
        else:
            print('Giving up...')
            raise ex
