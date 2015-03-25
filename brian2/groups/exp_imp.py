from brian2.groups.group import Group

class ImportExport(object):
	"""docstring for ImportExport"""
	def __init__(self, arg):
		super ImportExport, self).__init__()
		self.arg = arg

	@staticmethod
	def export_func1(vars,units = True):
		data = {}
		for var in vars:
			data[var] = np.array(Group().state(var,use_units = units, level= level+1),copy = True, subok = True)
		return data

	@staticmethod
	def export_func2(vars,units = True):
		old_data = {}
		for var in vars:

	        old_data[var] = np.array(Group().state(var, use_units=units,level=level+1),copy=True, subok=True)

	    data = pd.DataFrame(data = old_data); # Mentioning columns here was not necessary
	    return data






















'''
from brian2.groups.group import Group

class ImportExport(object):
		"""docstring for Impo"""
	def __init__(self, arg):

		super Impo, self).__init__()
		self.arg = arg


	@staticmethod
	def export_func1(vars,units = True):

    data = {}

       	for var in vars:

        	data[var] = np.array(Group().state(var, use_units=units,level=level+1),copy=True, subok=True)

       	return data

	@staticmethod
	def export_func2(vars,units = True):

	    old_data = {}

	    for var in vars:

	        old_data[var] = np.array(Group().state(var, use_units=units,level=level+1),copy=True, subok=True)

	    data = pd.DataFrame(data = old_data); # Mentioning columns here was not necessary
	    return data
		
'''


		
		