import numpy as np
from importexport import ImportExport

class DictImportExport(ImportExport):

    @property
    def name(self):
        return "dict"

    @staticmethod
    def export_data(group, variables, units=True, level=0):
        "export dict of numpy arrays"
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

    @property
    def name(self):
        return "pandas"

    @staticmethod
    def export_data(group, variables, units=True, level=0):
        "export pandas of numpy arrays"
        import pandas as pd
        data = DictImportExport.export_data(group, variables,
                                            units=units, level=level)
        return pd.DataFrame(data)

    @staticmethod
    def import_data(group, data, units=True, level=0):
        import pandas as pd
        colnames = data.columns
        array_data = data.as_matrix()
        for e, colname in enumerate(colnames):
            group.state(colname, use_units=units, level=level+1)[:] = array_data[:,e]
