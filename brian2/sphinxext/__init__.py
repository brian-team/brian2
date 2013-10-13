'''
Brian-specific extension to the Sphinx documentation generation system.
'''

def setup():
    '''
    Setup function for doctests (used by nosetest).
    We do not want to test this module's docstrings as not all installations
    have the necessary dependencies to build the documentation.
    '''
    from nose import SkipTest
    raise SkipTest('Do not test sphinx extension')