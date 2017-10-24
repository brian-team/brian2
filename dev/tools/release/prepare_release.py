import os

import brian2

from setversion import setversion, setreleasedate
# Ask for version number
print('Current version is: ' + brian2.__version__)
version = raw_input('Enter new Brian2 version number: ').strip()

# Set the version numbers
print 'Changing to new version', version
setversion(version)
setreleasedate()
print 'Done'

# generate the default preferences file
base, _ = os.path.split(brian2.__file__)
fname = os.path.join(base, 'default_preferences')
with open(fname, 'w') as f:
    f.write(brian2.prefs.as_file)

# commit
os.system('git commit -a -v -m "***** Release Brian2 %s *****"' % version)
# add tag
os.system('git tag -a -m "Release Brian2 %s" %s' % (version, version))

# print commands necessary for pushing
print('Review the last commit: ')
os.system('git show %s' % version)
print('')
print('To push, using the following command:')
print('git push --tags origin master')
