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
        # taken from get_states
        data = {}
        for var in variables:
            data[var] = np.array(group.state(var, use_units=units,
                                            level=level+1),
                                 copy=True, subok=True)
        return data

    @staticmethod
    def import_data(group, data, units=True, level=0):
        # taken from set_states
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
        import pandas as pd
        # we take adventage of already implemented exporter
        data = DictImportExport.export_data(group, variables,
                                            units=units, level=level)
        return pd.DataFrame(data)

    @staticmethod
    def import_data(group, data, units=True, level=0):
        # as pandas is not a default brian2 dependency we import it only in that namespace
        import pandas as pd
        colnames = data.columns
        array_data = data.as_matrix()
        for e, colname in enumerate(colnames):
            group.state(colname, use_units=units, level=level+1)[:] = array_data[:,e]
