import os

import brian2

basedir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

# Ask for version number
print('Current version is: ' + brian2.__version__)
version = input('Enter new Brian2 version number: ').strip()

# commit
os.system('git commit -a -v --allow-empty -m "***** Release Brian2 %s *****"' % version)
# add tag
os.system(f'git tag -a -m "Release Brian2 {version}" {version}')
# Run script to update codemeta.json
os.system(f'python {basedir}/.codemeta/create_codemeta.py')
# Include codemeta.json update in commit and update tag
os.system(f'git add {basedir}/codemeta.json && git commit --amend --no-edit && git tag -f -a -m "Release Brian2 {version}" {version}')

# print commands necessary for pushing
print('Review the last commit: ')
os.system('git show %s' % version)
print('')
print('To push, using the following command:')
print('git push --tags origin master')
