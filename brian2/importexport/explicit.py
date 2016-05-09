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
        return pd.DataFrame(data)

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
            group.state(colname, use_units=units, level=level+1)[:] = array_data[:, e]
