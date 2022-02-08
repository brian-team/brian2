import os

import brian2

# Ask for version number
print('Current version is: ' + brian2.__version__)
version = input('Enter new Brian2 version number: ').strip()

# commit
os.system('git commit -a -v --allow-empty -m "***** Release Brian2 %s *****"' % version)
# add tag
os.system('git tag -a -m "Release Brian2 %s" %s' % (version, version))

# print commands necessary for pushing
print('Review the last commit: ')
os.system('git show %s' % version)
print('')
print('To push, using the following command:')
print('git push --tags origin master')
