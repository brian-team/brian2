import numpy as np
from importexport import ImportExport

class DictImportExport(ImportExport):

    @staticmethod
    def export_data(group, variables, units=True, level=0):
        "export dict of numpy arrays"
        # taken from get_states
        data = {}
        for var in vars:
            data[var] = np.array(group.state(var, use_units=units,
                                            level=level+1),
                                 copy=True, subok=True)
        return data

    @staticmethod
    def import_data(group, data):
        # taken from set_states
        for key, value in data.iteritems():
            group.state(key, use_units=units, level=level+1)[:] = value
