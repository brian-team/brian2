'''
Module defining the `ImportExport` class that enables getting state variable
data in and out of groups in various formats (see `Group.get_states` and
`Group.set_states`).
'''
import abc
from abc import abstractmethod, abstractproperty


class ImportExport(object):
    '''
    Class for registering new import/export methods (via static methods). Also
    the base class that should be extended for such methods
    (`ImportExport.export_data`, `ImportExport.import_data`, and
    `ImportExport.name` have to be overwritten).

    See Also
    --------
    VariableOwner.get_states, VariableOwner.set_states

    '''
    __metaclass__ = abc.ABCMeta

    #: A dictionary mapping import/export methods names to `ImportExport` objects
    methods = dict()

    @staticmethod
    def register(importerexporter):
        '''
        Register a import/export method. Registered methods can be referred to
        via their name.

        Parameters
        ----------
        importerexporter : `ImportExport`
            The importerexporter object, e.g. an `DictImportExport`.
        '''
        if not isinstance(importerexporter, ImportExport):
            t = str(type(importerexporter))
            error_msg = ('Given importerexporter of type {} does not seem to '
                         'be a valid importerexporter.').format(t)
            raise ValueError(error_msg)
        name = importerexporter.name
        if name in ImportExport.methods:
            raise ValueError(('An import/export methods with the name {}'
                             'has already been registered').format(name))
        ImportExport.methods[name] = importerexporter

    @staticmethod
    @abstractmethod
    def export_data(group, variables):
        '''
        Asbtract static export data method with two obligatory parameters.
        It should return a copy of the current state variable values. The
        returned arrays are copies of the actual arrays that store the state
        variable values, therefore changing the values in the returned
        dictionary will not affect the state variables.

        Parameters
        ----------
        group : `Group`
            Group object.
        variables : list of str
            The names of the variables to extract.
        '''
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def import_data(group, data):
        '''
        Import and set state variables.

        Parameters
        ----------
        group : `Group`
            Group object.
        data : dict_like
            Data to import with variable names.
        '''
        raise NotImplementedError()

    @abstractproperty
    def name(self):
        '''
        Abstract property giving a method name.
        '''
        pass
