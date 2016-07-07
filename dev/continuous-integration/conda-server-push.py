import sys
import os
import glob
import time

from binstar_client.scripts.cli import main
from binstar_client.errors import BinstarError

token = os.environ['BINSTAR_TOKEN']
options = ['-t', token, 'upload',
           '-u', 'brian-team']
filenames = glob.glob('*.tar.bz2')
release = '+git' not in filenames[0]
if not release:
    options.extend(['--channel', 'dev', '--force'])

# Uploading sometimes fails due to server or network errors -- we try it five
# times before giving up
attempts = 5
uploaded = set()
for attempt in range(attempts):
    try:
        for filename in filenames:
            if filename in uploaded:  # We already uploaded this file
                continue
            main(args=options+[filename])
            uploaded.add(filename)
    except BinstarError as ex:
        print('Something did not work (%s).' % str(ex))
        if attempt < attempts - 1:
            print('Trying again in 10 seconds...')
            time.sleep(10)
        else:
            print('Giving up...')
            raise ex
