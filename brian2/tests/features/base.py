import brian2
import numpy
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

__all__ = ['FeatureTest', 'InaccuracyError', 'Configuration',
           'run_feature_tests', 'run_single_feature_test',
           'DefaultConfiguration', 'WeaveConfiguration',
           'CythonConfiguration', 'CPPStandaloneConfiguration',
           'CPPStandaloneConfigurationOpenMP']

class InaccuracyError(AssertionError):
    def __init__(self, error, *args):
        self.error = error
        AssertionError.__init__(self, *args)

class FeatureTest(object):
    '''
    '''
    
    category = None # a string with the category of features
    name = None # a string with the particular feature name within the category
    tags = None # a list of tags (strings) of features used

    @classmethod
    def fullname(cls):
        return cls.category+': '+cls.name
    
    def run(self):
        '''
        Runs the feature test but do not return results (some devices may 
        require an extra step before results are available).
        '''
        raise NotImplementedError
    
    def results(self):
        '''
        Return the results after a run call.
        '''
        raise NotImplementedError
    
    def compare(self, maxrelerr, results_base, results_test):
        '''
        Compare results from standard Brian run to another run.
        
        This method or `check` should be implemented.
        '''
        raise NotImplementedError
    
    def check(self, maxrelerr, results):
        '''
        Check results are valid (e.g. analytically).
        
        This method or `compare` should be implemented.
        '''
        raise NotImplementedError
    
    def compare_arrays(self, maxrelerr, v_base, v_test):
        '''
        Often you just want to compare the values of some arrays, this does that.
        '''
        err = numpy.amax(numpy.abs(v_base-v_test)/v_base)
        if err>maxrelerr:
            raise InaccuracyError(err)
    

class Configuration(object):
    '''
    '''
    
    name = None # The name of this configuration
    
    def before_run(self):
        pass
    
    def after_run(self):
        pass
    
    
class DefaultConfiguration(Configuration):
    name = 'Default'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.set_device('runtime')


class WeaveConfiguration(Configuration):
    name = 'Weave'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.set_device('runtime')
        brian2.prefs.codegen.target = 'weave'


class CythonConfiguration(Configuration):
    name = 'Cython'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.set_device('runtime')
        brian2.prefs.codegen.target = 'cython'
    
    
class CPPStandaloneConfiguration(Configuration):
    name = 'C++ standalone'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.set_device('cpp_standalone')
        
    def after_run(self):
        if os.path.exists('cpp_standalone'):
            shutil.rmtree('cpp_standalone')
        brian2.device.build(directory='cpp_standalone', compile=True, run=True,
                            with_output=False)


class CPPStandaloneConfigurationOpenMP(Configuration):
    name = 'C++ standalone (OpenMP)'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.set_device('cpp_standalone')
        brian2.prefs.devices.cpp_standalone.openmp_threads = 4
        
    def after_run(self):
        if os.path.exists('cpp_standalone'):
            shutil.rmtree('cpp_standalone')
        brian2.device.build(directory='cpp_standalone', compile=True, run=True,
                            with_output=False)
    
    
def results(configuration, feature):
    tempfilename = tempfile.mktemp('exception')
    code_string = '''
__file__ = '{fname}'
from {config_module} import {config_name}
from {feature_module} import {feature_name}
configuration = {config_name}()
feature = {feature_name}()
import warnings, traceback, pickle, sys, os
warnings.simplefilter('ignore')
try:
    configuration.before_run()
    feature.run()
    configuration.after_run()
    results = feature.results()
    f = open(r'{tempfname}', 'wb')
    pickle.dump(results, f, -1)
    f.close()
except Exception, ex:
    traceback.print_exc(file=sys.stdout)
    f = open(r'{tempfname}', 'wb')
    pickle.dump(ex, f, -1)
    f.close()
    '''.format(config_module=configuration.__module__,
               config_name=configuration.__name__,
               feature_module=feature.__module__,
               feature_name=feature.__name__,
               tempfname=tempfilename,
               fname=__file__,
               )
    args = [sys.executable, '-c',
            code_string]
    # Run the example in a new process and make sure that stdout gets
    # redirected into the capture plugin
    p = subprocess.Popen(args, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    #sys.stdout.write(stdout)
    #sys.stderr.write(stderr)
    f = open(tempfilename, 'rb')
    res = pickle.load(f)
    if isinstance(res, Exception):
        raise res
    else:
        return res
    

def check_or_compare(feature, res, baseline, maxrelerr):
    feature = feature()
    try:
        feature.check(maxrelerr, res)
    except NotImplementedError:
        feature.compare(maxrelerr, baseline, res)
        

def run_single_feature_test(configuration, feature):
    return results(configuration, feature)        

    
def run_feature_tests(configurations=None, feature_tests=None,
                      strict=1e-5, tolerant=0.05, verbose=True):
    if configurations is None:
        configurations = Configuration.__subclasses__()
    if feature_tests is None:
        feature_tests = FeatureTest.__subclasses__()
    if DefaultConfiguration in configurations:
        configurations.remove(DefaultConfiguration)
    configurations = [DefaultConfiguration]+configurations
    feature_tests.sort(key=lambda ft: ft.fullname())
    if verbose:
        print 'Running feature tests'
        print 'Configurations:', ', '.join(c.name for c in configurations)

    full_results = {}
    for ft in feature_tests:
        baseline = None
        if verbose:
            print ft.fullname()+': [',
        for configuration in configurations:
            txt = 'OK'
            sym = '.'
            exc = None
            try:
                res = results(configuration, ft)
                if configuration is DefaultConfiguration:
                    baseline = res
                    
                if isinstance(baseline, Exception):
                    sym = '?'
                    txt = 'Error in default configuration'
                else:
                    try:
                        check_or_compare(ft, res, baseline, strict)
                    except InaccuracyError as exc:
                        try:
                            check_or_compare(ft, res, baseline, tolerant)
                            sym = 'I'
                            txt = 'Poor (error=%.2f%%)' % (100.0*exc.error)
                        except InaccuracyError as exc:
                            sym = 'F'
                            txt = 'Fail (error=%.2f%%)' % (100.0*exc.error)
            except Exception as exc:
                res = exc
                sym = 'E'
                txt = 'Run failed.'
                if configuration is DefaultConfiguration:
                    raise
            sys.stdout.write(sym)
            full_results[configuration.name, ft.name] = (sym, txt, exc)
        if verbose:
            print ']'
        
    return FeatureTestResults(full_results, configurations, feature_tests)


class FeatureTestResults(object):
    def __init__(self, full_results, configurations, feature_tests):
        self.full_results = full_results
        self.configurations = configurations
        self.feature_tests = feature_tests
        
    @property
    def test_table(self):
        table = []
        table.append(['Test']+[c.name for c in self.configurations])
        curcat = ''
    
        for ft in self.feature_tests:
            cat = ft.category
            if cat!=curcat:
                table.append([cat]+['']*len(self.configurations))
                curcat = cat
            row = [ft.name]
            for configuration in self.configurations:
                sym, txt, exc = self.full_results[configuration.name, ft.name]
                row.append(txt)
            table.append(row)
        return make_table(table)
    
    @property
    def tag_table(self):
        return 'TODO'
        
    def __str__(self):
        r = ''
        s = 'Feature test results'
        r += s+'\n'+'-'*len(s)+'\n\n'+self.test_table+'\n'
        s = 'Tag results'
        r += s+'\n'+'-'*len(s)+'\n\n'+self.tag_table+'\n'
        return r
    __repr__ = __str__

# Code below auto generates restructured text tables, copied from:
# http://stackoverflow.com/questions/11347505/what-are-some-approaches-to-outputting-a-python-data-structure-to-restructuredte

def make_table(grid):
    max_cols = [max(out) for out in map(list, zip(*[[len(item) for item in row] for row in grid]))]
    rst = table_div(max_cols, 1)

    for i, row in enumerate(grid):
        header_flag = False
        if i == 0 or i == len(grid)-1: header_flag = True
        rst += normalize_row(row,max_cols)
        rst += table_div(max_cols, header_flag )
    return rst

def table_div(max_cols, header_flag=1):
    out = ""
    if header_flag == 1:
        style = "="
    else:
        style = "-"

    for max_col in max_cols:
        out += max_col * style + " "

    out += "\n"
    return out


def normalize_row(row, max_cols):
    r = ""
    for i, max_col in enumerate(max_cols):
        r += row[i] + (max_col  - len(row[i]) + 1) * " "

    return r + "\n"
