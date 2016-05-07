import abc
from abc import abstractmethod, abstractproperty

class ImportExport(object):
    __metaclass__ = abc.ABCMeta
    
    #: A dictionary mapping import/export methods names to `ImportExport` objects
    methods = dict()

    @staticmethod    
    def register(name, importerexporter):
        '''
        Register a import/export method. Registered importerexporter can be referred to
        via their name.
        
        Parameters
        ----------
        name : str
            A short name for the import/export class (e.g. `'dict'`)
        importerexporter : `ImportExport`
            The importerexporter object, e.g. an `DictImportExport`.
        '''
        
        name = name.lower()
        if name in ImportExport.methods:
            raise ValueError(("An import/export methods with the name {}" +\
                             "has already been registered").format(name))

        if not isinstance(importerexporter, ImportExport):
            raise ValueError(('Given importerexporter of type {} does not seem to ' +\
                              'be a valid importerexporter.').format(str(type(importerexporter))))
        ImportExport.methods[name] = importerexporter

    @staticmethod
    @abstractmethod
    def export_data(group, variables):
        '''
        Asbtract static export data method with two obligatory parameters.
        It should return a copy of the current state variable values. The returned arrays
        are copies of the actual arrays that store the state variable values,
        therefore changing the values in the returned dictionary will not affect
        the state variables.
        
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

