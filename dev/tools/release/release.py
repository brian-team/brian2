import datetime
import os
import sys

import brian2

basedir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)

# Ask for version number
print("Current version is: " + brian2.__version__)
version = input("Enter new Brian2 version number: ").strip()

# Run script to update codemeta.json
os.system(f"python {basedir}/.codemeta/create_codemeta.py {version}")

# Set version and release date in CITATION.cff
new_lines = []
with open(f"{basedir}/CITATION.cff") as f:
    for line in f:
        if line.startswith("version: "):
            line = f"version: '{version}'\n"
        if line.startswith("date-released: "):
            line = f"date-released: '{datetime.date.today().isoformat()}'\n"
        new_lines.append(line)
new_lines.append("")  # Add empty line at the end
with open(f"{basedir}/CITATION.cff", "w") as f:
    f.writelines(new_lines)

# commit
x = os.system(
    f'git add {basedir}/codemeta.json && git add {basedir}/CITATION.cff && git commit -m "***** Release Brian2 {version} *****"'
)
if x:
    print("Making commit failed with error code: ", x)
    sys.exit(x)

# add tag
os.system(f'git tag -a -m "Release Brian2 {version}" {version}')

# print commands necessary for pushing
print("Review the last commit: ")
os.system("git show %s" % version)
print("")
print("To push, using the following command:")
print("git push --tags origin master")
