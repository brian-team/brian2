__all__ = ['FeatureTest', 'InaccuracyError', 'Configuration',
           'run_feature_tests']

class InaccuracyError(Exception):
    pass

class FeatureTest(object):
    '''
    '''
    
    category = None # a string with the category of features
    name = None # a string with the particular feature name within the category
    tags = None # a list of tags (strings) of features used
    
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
    
    def compare(self, results_base, results_test):
        '''
        Compare results from standard Brian run to another run.
        
        This method or `check` should be implemented.
        '''
        raise NotImplementedError
    
    def check(self, results):
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
    name = 'Brian defaults'
    
    
def results(configuration, feature):
    configuration = configuration()
    feature = feature()
    configuration.before_run()
    feature.run()
    configuration.after_run()
    return feature.results()
    
    
def run_feature_tests(configurations=None, feature_tests=None):
    if configurations is None:
        configurations = Configuration.__subclasses__()
    if feature_tests is None:
        feature_tests = FeatureTest.__subclasses__()
    print 'Running feature tests'
    print 'Configurations:', ', '.join(c.name for c in configurations)
    print 'Feature tests:', ', '.join(ft.name for ft in feature_tests)
    all_results = {}
    for configuration in configurations:
        print 'Starting configuration:', configuration.name
        for ft in feature_tests:
            try:
                res = results(configuration, ft)
                sym = '.'
            except InaccuracyError as exc:
                res = exc
                sym = 'I'
            except Exception as exc:
                res = exc
                sym = 'F'
            all_results[configuration.name, ft.name] = res
            #print sym,
            print 'Test', ft.name, 'result:', sym
