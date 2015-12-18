import brian2
import numpy
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import itertools
import re

from brian2.utils.stringtools import indent

from collections import defaultdict

__all__ = ['FeatureTest', 'SpeedTest',
           'InaccuracyError', 'Configuration',
           'run_feature_tests', 'run_single_feature_test',
           'run_speed_tests',
           'DefaultConfiguration', 'LocalConfiguration',
           'NumpyConfiguration', 'WeaveConfiguration',
           'CythonConfiguration', 'CPPStandaloneConfiguration',
           'CPPStandaloneConfigurationOpenMP']


class InaccuracyError(AssertionError):
    def __init__(self, error, *args):
        self.error = error
        AssertionError.__init__(self, *args)

class BaseTest(object):
    '''
    '''
    
    category = None # a string with the category of features
    name = None # a string with the particular feature name within the category
    tags = None # a list of tags (strings) of features used
    # whether or not to allow the device to override the time: this can be used to remove the
    # compilation overheads on certain devices (but some tests might want to include this)
    allow_time_override = True

    @classmethod
    def fullname(cls):
        return cls.category+': '+cls.name

    def run(self):
        '''
        Runs the feature test but do not return results (some devices may 
        require an extra step before results are available).
        '''
        raise NotImplementedError

    def timed_run(self, duration):
        '''
        Do a timed run. This means that for RuntimeDevice it will run for defaultclock.dt before running for the
        rest of the duration. This means total run duration will be duration+defaultclock.dt.
        For standalone devices, this feature may or may not be implemented.
        '''
        if isinstance(brian2.get_device(), brian2.devices.RuntimeDevice):
            brian2.run(brian2.defaultclock.dt, level=1)
            brian2.run(duration, level=1)
        else:
            brian2.run(duration, level=1)


class FeatureTest(BaseTest):
    '''
    '''
        
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
        if isinstance(v_base, dict):
            for k in v_base.keys():
                self.compare_arrays(maxrelerr, v_base[k], v_test[k])
        else:
            I = (v_base!=0)
            err = numpy.amax(numpy.abs(v_base[I]-v_test[I])/v_base[I])
            if err>maxrelerr:
                raise InaccuracyError(err)
            if (v_test[-I]!=0).any():
                raise InaccuracyError(numpy.inf)
      
        
class SpeedTest(BaseTest):
    n_range = [1]
    n_label = 'n'
    n_axis_log = True
    time_axis_log = True
    
    def __init__(self, n):
        self.n = n
            
    def results(self):
        return self.n
    
    def compare(self, maxrelerr, results_base, results_test):
        pass
    
    def check(self, maxrelerr, results):
        pass
    
    def __call__(self):
        return self


class Configuration(object):
    '''
    '''
    
    name = None # The name of this configuration

    def __init__(self, maximum_run_time=1e7*brian2.second):
        maximum_run_time = float(maximum_run_time)*brian2.second
        self.maximum_run_time = maximum_run_time

    def before_run(self):
        pass
    
    def after_run(self):
        pass
    
    def get_last_run_time(self):
        '''
        Implement this to overwrite the measured runtime (e.g. to remove overhead).
        '''
        if hasattr(brian2.device, '_last_run_time'):
            return brian2.device._last_run_time
        raise NotImplementedError

    def get_last_run_completed_fraction(self):
        '''
        Implement this to overwrite the amount of the last run that was completed (for devices that allow breaking
        early if the maximum run time is exceeded).
        '''
        if hasattr(brian2.device, '_last_run_completed_fraction'):
            return brian2.device._last_run_completed_fraction
        return 1.0

    
class DefaultConfiguration(Configuration):
    name = 'Default'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.set_device('runtime')


class LocalConfiguration(Configuration):
    name = 'Local'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.set_device('runtime')
        brian2.prefs.load_preferences()


class NumpyConfiguration(Configuration):
    name = 'Numpy'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.set_device('runtime')
        brian2.prefs.codegen.target = 'numpy'


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
        brian2.set_device('cpp_standalone', build_on_run=False)
        
    def after_run(self):
        if os.path.exists('cpp_standalone'):
            shutil.rmtree('cpp_standalone')
        brian2.device.build(directory='cpp_standalone', compile=True, run=True,
                            with_output=False)

class CPPStandaloneConfigurationOpenMP(Configuration):
    name = 'C++ standalone (OpenMP)'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.set_device('cpp_standalone', build_on_run=False)
        brian2.prefs.devices.cpp_standalone.openmp_threads = 4
        
    def after_run(self):
        if os.path.exists('cpp_standalone'):
            shutil.rmtree('cpp_standalone')
        brian2.device.build(directory='cpp_standalone', compile=True, run=True,
                            with_output=False)
    
    
def results(configuration, feature, n=None, maximum_run_time=1e7*brian2.second):
    tempfilename = tempfile.mktemp('exception')
    if n is None:
        init_args = ''
    else:
        init_args = str(n)
    code_string = '''
__file__ = '{fname}'
import brian2
from {config_module} import {config_name}
from {feature_module} import {feature_name}
configuration = {config_name}()
feature = {feature_name}({init_args})
import warnings, traceback, pickle, sys, os, time
warnings.simplefilter('ignore')
try:
    start_time = time.time()
    configuration.before_run()
    brian2.device._set_maximum_run_time({maximum_run_time})
    feature.run()
    configuration.after_run()
    results = feature.results()
    run_time = time.time()-start_time
    if feature.allow_time_override:
        try:
            run_time = configuration.get_last_run_time()
        except NotImplementedError:
            pass
    lrcf = configuration.get_last_run_completed_fraction()
    run_time = run_time/lrcf
    prof_info = brian2.magic_network.profiling_info
    new_prof_info = []
    for n, t in prof_info:
        new_prof_info.append((n, t/lrcf))
    f = open(r'{tempfname}', 'wb')
    pickle.dump((None, results, run_time, new_prof_info), f, -1)
    f.close()
except Exception, ex:
    #traceback.print_exc(file=sys.stdout)
    tb = traceback.format_exc()
    f = open(r'{tempfname}', 'wb')
    pickle.dump((tb, ex, 0.0, []), f, -1)
    f.close()
    '''.format(config_module=configuration.__module__,
               config_name=configuration.__name__,
               feature_module=feature.__module__,
               feature_name=feature.__name__,
               tempfname=tempfilename,
               fname=__file__,
               init_args=init_args,
               maximum_run_time=float(maximum_run_time),
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
    tb, res, runtime, profiling_info = pickle.load(f)
    return tb, res, runtime, profiling_info
    

def check_or_compare(feature, res, baseline, maxrelerr):
    feature = feature()
    try:
        feature.check(maxrelerr, res)
    except NotImplementedError:
        feature.compare(maxrelerr, baseline, res)
        

def run_single_feature_test(configuration, feature):
    return results(configuration, feature)        

    
def run_feature_tests(configurations=None, feature_tests=None,
                      strict=1e-5, tolerant=0.05, verbose=True, maximum_run_time=1e7*brian2.second):
    if configurations is None:
        # some configurations to attempt to import
        try:
            import brian2genn.correctness_testing
        except:
            pass
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
    tag_results = defaultdict(lambda:defaultdict(list))
    for ft in feature_tests:
        baseline = None
        if verbose:
            print ft.fullname()+': [',
        for configuration in configurations:
            txt = 'OK'
            sym = '.'
            exc = None
            tb, res, runtime, prof_info = results(configuration, ft, maximum_run_time=maximum_run_time)
            if isinstance(res, Exception):
                if isinstance(res, NotImplementedError):
                    sym = 'N'
                    txt = 'Not implemented'
                else:
                    sym = 'E'
                    txt = 'Error'
                if configuration is DefaultConfiguration:
                    raise res
            else:
                if configuration is DefaultConfiguration:
                    baseline = res                    
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
            sys.stdout.write(sym)
            full_results[configuration.name, ft.fullname()] = (sym, txt, exc, tb, runtime, prof_info)
            for tag in ft.tags:
                tag_results[tag][configuration.name].append((sym, txt, exc, tb, runtime, prof_info))
        if verbose:
            print ']'
        
    return FeatureTestResults(full_results, tag_results,
                              configurations, feature_tests)


class FeatureTestResults(object):
    def __init__(self, full_results, tag_results,
                 configurations, feature_tests):
        self.full_results = full_results
        self.tag_results = tag_results
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
                sym, txt, exc, tb, runtime, prof_info = self.full_results[configuration.name,
                                                                          ft.fullname()]
                row.append(txt)
            table.append(row)
        return make_table(table)
    
    @property
    def tag_table(self):
        table = []
        table.append(['Tag']+[c.name for c in self.configurations])
        tags = sorted(self.tag_results.keys())
    
        for tag in tags:
            row = [tag]
            for configuration in self.configurations:
                tag_res = self.tag_results[tag][configuration.name]
                syms = [sym for sym, txt, exc, tb, runtime, prof_info in tag_res]
                n = len(syms)
                okcount = sum(sym=='.' for sym in syms)
                poorcount = sum(sym=='I' for sym in syms)
                failcount = sum(sym=='F' for sym in syms)
                errcount = sum(sym=='E' for sym in syms)
                nicount = sum(sym=='N' for sym in syms)
                if okcount==n:
                    txt = 'OK'
                elif nicount==n:
                    txt = 'Not implemented'
                elif errcount==n:
                    txt = 'Unsupported'
                elif okcount+poorcount==n:
                    txt = 'Poor (%d%%)' % int(poorcount*100.0/n)
                elif okcount+poorcount+failcount==n:
                    txt = 'Fail: {fail}% (poor={poor}%)'.format(
                        fail=int(failcount*100.0/n),
                        poor=int(poorcount*100.0/n),
                        )
                else:
                    txt = 'Fail: OK={ok}%, Poor={poor}%, Fail={fail}%, NotImpl={ni}% Error={err}%'.format(
                        ok=int(okcount*100.0/n), poor=int(poorcount*100.0/n), 
                        fail=int(failcount*100.0/n), err=int(errcount*100.0/n),
                        ni=int(nicount*100.0/n), 
                        )
                row.append(txt)
            table.append(row)
        return make_table(table)
    
    @property
    def tables(self):
        r = ''
        s = 'Feature test results'
        r += s+'\n'+'-'*len(s)+'\n\n'+self.test_table+'\n'
        s = 'Tag results'
        r += s+'\n'+'-'*len(s)+'\n\n'+self.tag_table+'\n'
        return r
    
    @property
    def exceptions(self):
        exc_list = []
        for configuration in self.configurations:
            curconfig = []
            for ft in self.feature_tests:
                sym, txt, exc, tb, runtime, prof_info = self.full_results[configuration.name,
                                                                          ft.fullname()]
                if tb is not None:
                    curconfig.append((ft.fullname(), tb))
            if len(curconfig):
                exc_list.append((configuration.name, curconfig))
        if len(exc_list)==0:
            return ''
        r = ''
        s = 'Exceptions'
        r += s+'\n'+'-'*len(s)+'\n\n'
        for config_name, curconfig in exc_list:
            s = config_name
            r += s+'\n'+'^'*len(s)+'\n\n'
            for name, tb in curconfig:
                r += name+'::\n\n'+indent(tb)+'\n\n' 
        return r
    
    @property
    def tables_and_exceptions(self):
        return self.tables+'\n'+self.exceptions
        
    def __str__(self):
        return self.tables
    __repr__ = __str__


def run_speed_tests(configurations=None, speed_tests=None, run_twice=True, verbose=True,
                    n_slice=slice(None), maximum_run_time=1e7*brian2.second):
    if configurations is None:
        # some configurations to attempt to import
        try:
            import brian2genn.correctness_testing
        except:
            pass
        configurations = Configuration.__subclasses__()
    if speed_tests is None:
        speed_tests = SpeedTest.__subclasses__()
    speed_tests.sort(key=lambda ft: ft.fullname())
    if verbose:
        print 'Running speed tests'
        print 'Configurations:', ', '.join(c.name for c in configurations)

    full_results = {}
    tag_results = defaultdict(lambda:defaultdict(list))
    for ft in speed_tests:
        if verbose:
            print ft.fullname()+': ',
        for n in ft.n_range[n_slice]:
            if verbose:
                print 'n=%d [' % n,
            for configuration in configurations:
                sym = '.'
                for _ in xrange(1+int(run_twice)):
                    tb, res, runtime, prof_info = results(configuration, ft, n, maximum_run_time=maximum_run_time)
                if isinstance(res, Exception):
                    if isinstance(res, NotImplementedError):
                        sym = 'N'
                    else:
                        sym = 'E'
                    if configuration is DefaultConfiguration:
                        raise res
                    runtime = numpy.NAN
                sys.stdout.write(sym)
                full_results[configuration.name, ft.fullname(), n, 'All'] = runtime
                suffixtime = defaultdict(float)
                overheadstime = float(runtime)
                for codeobjname, proftime in prof_info:
                    # parts = codeobjname.split('_')
                    # parts = [part for part in parts if not re.match(r'\d+', part)]
                    #suffix = '_'.join(parts)
                    suffix = codeobjname
                    suffixtime[suffix] += proftime
                    overheadstime -= float(proftime)
                for suffix, proftime in suffixtime.items():
                    full_results[configuration.name, ft.fullname(), n, suffix] = proftime
                full_results[configuration.name, ft.fullname(), n, 'Overheads'] = overheadstime
            if verbose:
                print ']',
        if verbose:
            print
        
    return SpeedTestResults(full_results, configurations, speed_tests)


class SpeedTestResults(object):
    def __init__(self, full_results, configurations, speed_tests):
        self.full_results = full_results
        self.configurations = configurations
        self.speed_tests = speed_tests
        
    def get_ns(self, fullname):
        L = [(cn, fn, n, s) for cn, fn, n, s in self.full_results.keys() if fn==fullname]
        confignames, fullnames, n, codeobjsuffixes  = zip(*L)
        return numpy.array(sorted(list(set(n))))

    def get_codeobjsuffixes(self, fullname):
        L = [(cn, fn, n, s) for cn, fn, n, s in self.full_results.keys() if fn==fullname]
        confignames, fullnames, n, codeobjsuffixes  = zip(*L)
        return set(codeobjsuffixes)

    def plot_all_tests(self, relative=False, profiling_minimum=1.0):
        if relative and profiling_minimum<1:
            raise ValueError("Cannot use relative plots with profiling")
        import pylab
        for st in self.speed_tests:
            fullname = st.fullname()
            pylab.figure()
            ns = self.get_ns(fullname)
            codeobjsuffixes = self.get_codeobjsuffixes(fullname)
            codeobjsuffixes.remove('All')
            codeobjsuffixes.remove('Overheads')
            codeobjsuffixes = ['All', 'Overheads']+sorted(codeobjsuffixes)
            if relative or profiling_minimum==1:
                codeobjsuffixes = ['All']
            baseline = None
            havelabel = set()
            markerstyles_cycle = iter(itertools.cycle(['o', 's', 'd', 'v', 'p', 'h', '^', '<', '>']))
            dashes = {}
            markerstyles = {}
            for isuffix, suffix in enumerate(codeobjsuffixes):
                cols = itertools.cycle(pylab.rcParams['axes.color_cycle'])
                for (iconfig, config), col in zip(enumerate(self.configurations), cols):
                    configname = config.name
                    runtimes = []
                    skip = True
                    for n in ns:
                        runtime = self.full_results.get((configname, fullname, n, 'All'), numpy.nan)
                        thistime = self.full_results.get((configname, fullname, n, suffix), numpy.nan)
                        if float(thistime/runtime)>=profiling_minimum:
                            skip = False
                        runtimes.append(thistime)
                    if skip:
                        continue
                    runtimes = numpy.array(runtimes)
                    if relative:
                        if baseline is None:
                            baseline = runtimes
                        runtimes = baseline/runtimes
                    if suffix=='All':
                        lw = 2
                        label = configname
                    else:
                        lw = 1
                        label = suffix
                    plottable = sum(-numpy.isnan(runtimes[1:]+runtimes[:-1]))
                    if plottable:
                        if label in havelabel:
                            label = None
                        else:
                            havelabel.add(label)
                        dash = None
                        msty = None
                        if suffix!='All':
                            if suffix in dashes:
                                dash = dashes[suffix]
                                msty = markerstyles[suffix]
                            else:
                                j = len(dashes)
                                dash = (8, 2)
                                for b in bin(j)[2:]:
                                    if b=='0':
                                        dash = dash+(2, 2)
                                    else:
                                        dash = dash+(4, 2)
                                dashes[suffix] = dash
                                markerstyles[suffix] = msty = markerstyles_cycle.next()
                        line = pylab.plot(ns, runtimes, lw=lw, color=col, marker=msty,
                                          mec='none', ms=8, label=label)[0]
                        if dash is not None:
                            line.set_dashes(dash)
            pylab.title(fullname)
            pylab.legend(loc='best', fontsize='x-small', handlelength=8.0)
            pylab.xlabel(st.n_label)
            if st.n_axis_log:
                pylab.gca().set_xscale('log')
            if st.time_axis_log:
                pylab.gca().set_yscale('log')

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
