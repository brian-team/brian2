import sys
import os
import yaml
import glob
import shutil

from conda_build.config import get_or_merge_config

with open(os.path.join(sys.argv[1], 'meta.yaml')) as f:
    name = yaml.load(f)['package']['name']

packages_dir = get_or_merge_config(None).bldpkgs_dir

binary_package_glob = os.path.join(packages_dir, '{0}*.tar.bz2'.format(name))
binary_packages = glob.glob(binary_package_glob)
for binary_package in binary_packages:
    shutil.move(binary_package, '.')
