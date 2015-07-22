import os
import glob
import subprocess
import traceback

token = os.environ['BINSTAR_TOKEN']
options = ['-t', token, 'upload',
           '-u', 'brian-team']
filename = glob.glob('*.tar.bz2')
assert len(filename) == 1, 'Expected to find one .tar.bz2 file, found %d' % len(filename)
release = '+git' not in filename[0]
if not release:
    options.extend(['--channel', 'dev', '--force'])
cmd = ['conda-server'] + options
cmd.extend(filename)

try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError:
    traceback.print_exc()
