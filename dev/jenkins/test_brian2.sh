source /home/jenkins/.jenkins/virtual_envs/$PythonVersion/$packages/bin/activate 
# get the newest version of nose and coverage, ignoring installed packages
pip install --upgrade -I nose coverage || :

# Make sure pyparsing and ipython (used for pretty printing) are installed
if [ ${PythonVersion:0:1} == '2' ]; then
	pip install pyparsing==1.5.7
	else
	pip install pyparsing --upgrade
fi
pip install ipython

# Make sure we have sphinx (for testing the sphinxext)
pip install sphinx

echo "Using newest available package versions"

# Use numpy 1.6.2 for Python 2 for now
if [ ${PythonVersion:0:1} == '2' ]; then
	pip install numpy==1.6.2
	else
	pip install numpy --upgrade
fi
 
pip install --upgrade scipy
pip install --upgrade sympy
pip install --upgrade matplotlib

# Print the version numbers for the dependencies
python -c "import numpy; print 'numpy version: ', numpy.__version__"
python -c "import scipy; print 'scipy version: ', scipy.__version__"
python -c "import sympy; print 'sympy version: ', sympy.__version__"
python -c "import matplotlib; print 'matplotlib version: ', matplotlib.__version__"

# Build Brian2
python setup.py build --build-lib=build/lib
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH

# delete remaining compiled code from previous runs
echo deleting '~/.python*_compiled' if it exists
rm -r ~/.python*_compiled || :

# Run unit tests and record coverage but do not fail the build if anything goes wrong here
coverage erase --rcfile=.coveragerc || :
coverage run --rcfile=.coveragerc ~/.jenkins/virtual_envs/$PythonVersion/$packages/bin/nosetests --with-xunit --logging-clear-handlers --verbose --with-doctest brian2 || :
coverage xml --rcfile=.coveragerc || :
