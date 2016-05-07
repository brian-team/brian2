import abc
from abc import abstractmethod, abstractproperty

class ImportExport(object):
    __metaclass__ = abc.ABCMeta

    methods = dict()

    @staticmethod    
    def register(name, importerexporter):
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
        "export function"
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def import_data(group, data):
        raise NotImplementedError()

    @abstractproperty
    def name(self):
        pass

