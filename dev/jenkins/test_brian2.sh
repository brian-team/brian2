source /home/jenkins/.jenkins/virtual_envs/$PythonVersion/$packages/bin/activate
# get the newest version of nose and coverage, ignoring installed packages
pip install --upgrade -I nose coverage || :

# Make sure pyparsing and ipython (used for pretty printing) are installed
pip install pyparsing --upgrade
pip install ipython

# Make sure we have sphinx (for testing the sphinxext)
pip install sphinx

echo "Using newest available package versions"

pip install --upgrade numpy
pip install --upgrade scipy
pip install --upgrade sympy
pip install --upgrade matplotlib

# Print the version numbers for the dependencies
python -c "import numpy; print('numpy version: ' + numpy.__version__)"
python -c "import scipy; print('scipy version: ' + scipy.__version__)"
python -c "import sympy; print('sympy version: ' + sympy.__version__)"
python -c "import matplotlib; print('matplotlib version: ' + matplotlib.__version__)"

# Build Brian2
python setup.py build --build-lib=build/lib

# Move to the build directory to make sure it is used for testing
# (important for Python 3)
cd build/lib

# delete remaining compiled code from previous runs
echo deleting '~/.python*_compiled' if it exists
rm -r ~/.python*_compiled || :

# Run unit tests and record coverage but do not fail the build if anything goes wrong here
coverage erase --rcfile=../../.coveragerc || :
coverage run --rcfile=../../.coveragerc ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/nosetests --with-xunit --logging-clear-handlers --verbose --with-doctest brian2 || :
coverage xml -i --rcfile=../../.coveragerc || :
