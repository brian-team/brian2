import numpy as np
from importexport import ImportExport

from brian2.units.fundamentalunits import Quantity, get_unit

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
            if getattr(group.variables[key],'read_only'):
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
        # as pandas is not a default brian2 dependency we import it only in that namespace
        try:
            import pandas as pd
        except ImportError as ex:
            raise ImportError('Exporting to pandas needs a working installation of pandas. '
                              'Importing pandas failed: ' + str(ex))
        # we take adventage of already implemented exporter
        data = DictImportExport.export_data(group, variables,
                                            units=units, level=level)
        pandas_data = pd.DataFrame(data)
        # above operation looses information about units so we add it to property
        if units:
            pandas_data.units = [get_unit(data[col]) if type(data[col])==Quantity else None
                                                     for col in pandas_data.columns]
        return pandas_data

    @staticmethod
    def import_data(group, data, units=True, level=0):
        # as pandas is not a default brian2 dependency we import it only in that namespace
        try:
            import pandas as pd
        except ImportError as ex:
            raise ImportError('Exporting to pandas needs a working installation of pandas. '
                              'Importing pandas failed: ' + str(ex))
        colnames = data.columns
        array_data = data.as_matrix()
        for e, colname in enumerate(colnames):
            if getattr(group.variables[colname],'read_only'):
                raise TypeError('Variable {} is read-only.'.format(colname))
            if units:
                if data.units[e] is not None:
                    group.state(colname, use_units=units, level=level+1)[:] = np.squeeze(array_data[:, e]) * data.units[e]
                    continue
                group.state(colname, use_units=units, level=level+1)[:] = np.squeeze(array_data[:, e])
            else:
                group.state(colname, use_units=units, level=level+1)[:] = np.squeeze(array_data[:, e])
