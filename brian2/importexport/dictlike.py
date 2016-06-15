'''
Module providing `DictImportExport` and `PandasImportExport` (requiring a
working installation of pandas).
'''
import numpy as np
from importexport import ImportExport


class DictImportExport(ImportExport):
    '''
    An importer/exporter for variables in format of dict of numpy arrays.
    '''
    @property
    def name(self):
        return "dict"

    @staticmethod
    def export_data(group, variables, units=True, level=0):
        data = {}
        for var in variables:
            data[var] = np.array(group.state(var, use_units=units,
                                             level=level+1),
                                 copy=True, subok=True)
        return data

    @staticmethod
    def import_data(group, data, units=True, level=0):
        for key, value in data.iteritems():
            if getattr(group.variables[key], 'read_only'):
                raise TypeError('Variable {} is read-only.'.format(key))
            group.state(key, use_units=units, level=level+1)[:] = value


class PandasImportExport(ImportExport):
    '''
    An importer/exporter for variables in pandas DataFrame format.
    '''

    @property
    def name(self):
        return "pandas"

    @staticmethod
    def export_data(group, variables, units=True, level=0):
        # pandas is not a default brian2 dependency, only import it here
        try:
            import pandas as pd
        except ImportError as ex:
            raise ImportError('Exporting to pandas needs a working installation'
                              ' of pandas. Importing pandas failed: ' + str(ex))
        if units:
            raise NotImplementedError('Units not supported when exporting to '
                                      'pandas data frame')
        # we take advantage of the already implemented exporter
        data = DictImportExport.export_data(group, variables,
                                            units=units, level=level)
        pandas_data = pd.DataFrame(data)
        return pandas_data

    @staticmethod
    def import_data(group, data, units=True, level=0):
        # pandas is not a default brian2 dependency, only import it here
        try:
            import pandas as pd
        except ImportError as ex:
            raise ImportError('Exporting to pandas needs a working installation'
                              ' of pandas. Importing pandas failed: ' + str(ex))
        if units:
            raise NotImplementedError('Units not supported when importing from '
                                      'pandas data frame')
        colnames = data.columns
        array_data = data.values
        for e, colname in enumerate(colnames):
            if getattr(group.variables[colname], 'read_only'):
                raise TypeError('Variable {} is read-only.'.format(colname))
            state = group.state(colname, use_units=units, level=level+1)
            state[:] = np.squeeze(array_data[:, e])
