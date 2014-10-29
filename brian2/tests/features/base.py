import brian2
import os
import shutil
import sys
from StringIO import StringIO

__all__ = ['FeatureTest', 'InaccuracyError', 'Configuration',
           'run_feature_tests',
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
        brian2.prefs.read_preference_file(StringIO(brian2.prefs.defaults_as_file))
        brian2.set_device('runtime')


class WeaveConfiguration(Configuration):
    name = 'Weave'
    def before_run(self):
        brian2.prefs.read_preference_file(StringIO(brian2.prefs.defaults_as_file))
        brian2.set_device('runtime')
        brian2.prefs.codegen.target = 'weave'


class CythonConfiguration(Configuration):
    name = 'Cython'
    def before_run(self):
        brian2.prefs.read_preference_file(StringIO(brian2.prefs.defaults_as_file))
        brian2.set_device('runtime')
        brian2.prefs.codegen.target = 'cython'
    
    
class CPPStandaloneConfiguration(Configuration):
    name = 'C++ standalone'
    def before_run(self):
        brian2.prefs.read_preference_file(StringIO(brian2.prefs.defaults_as_file))
        brian2.set_device('cpp_standalone')
        
    def after_run(self):
        if os.path.exists('cpp_standalone'):
            shutil.rmtree('cpp_standalone')
        brian2.device.build(directory='cpp_standalone', compile=True, run=True)


class CPPStandaloneConfigurationOpenMP(Configuration):
    name = 'C++ standalone (OpenMP)'
    def before_run(self):
        brian2.prefs.read_preference_file(StringIO(brian2.prefs.defaults_as_file))
        brian2.set_device('cpp_standalone')
        brian2.prefs.codegen.cpp_standalone.openmp_threads = 4
        
    def after_run(self):
        if os.path.exists('cpp_standalone'):
            shutil.rmtree('cpp_standalone')
        brian2.device.build(directory='cpp_standalone', compile=True, run=True)
    
    
def results(configuration, feature):
    configuration.before_run()
    feature.run()
    configuration.after_run()
    return feature.results()


def check_or_compare(feature, res, baseline, maxrelerr):
    try:
        feature.check(maxrelerr, res)
    except NotImplementedError:
        feature.compare(maxrelerr, baseline, res)
    
    
def run_feature_tests(configurations=None, feature_tests=None,
                      strict=1e-5, tolerant=0.05):
    if configurations is None:
        configurations = Configuration.__subclasses__()
    if feature_tests is None:
        feature_tests = FeatureTest.__subclasses__()
    if DefaultConfiguration in configurations:
        configurations.remove(DefaultConfiguration)
    configurations = [DefaultConfiguration]+configurations
    feature_tests.sort(key=lambda ft: ft.fullname())
    print 'Running feature tests'
    print 'Configurations:', ', '.join(c.name for c in configurations)
    print 'Feature tests:', ', '.join(ft.fullname() for ft in feature_tests)

    table = []
    table.append(['Test']+[c.name for c in configurations])
    curcat = ''

    for ftc in feature_tests:
        cat = ftc.category
        if cat!=curcat:
            table.append([cat]+['']*len(configurations))
            curcat = cat
        row = [ftc.name]
        baseline = None
        for configurationc in configurations:
            configuration = configurationc()
            ft = ftc()
            txt = 'OK'
            sym = '.'
            try:
                res = results(configuration, ft)
                if configurationc is DefaultConfiguration:
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
                raise
            sys.stdout.write(sym)
            row.append(txt)
        table.append(row)
    print
    return make_table(table)
            

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