import sys
import os
import yaml
import glob
import shutil
try:
    from conda_build.config import config
except ImportError:
    # For older versions of conda-build
    from conda_build import config

with open(os.path.join(sys.argv[1], 'meta.yaml')) as f:
    name = yaml.load(f)['package']['name']

binary_package_glob = os.path.join(config.bldpkgs_dir, '{0}*.tar.bz2'.format(name))
binary_packages = glob.glob(binary_package_glob)
for binary_package in binary_packages:
    shutil.move(binary_package, '.')
