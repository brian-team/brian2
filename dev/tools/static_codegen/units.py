import os, re

curdir, _ = os.path.split(__file__)
units_fname = os.path.normpath(os.path.join(curdir, '../../../brian2/units/allunits.py'))

_siprefixes = {"y":1e-24, "z":1e-21, "a":1e-18, "f":1e-15, "p":1e-12, "n":1e-9,
               "u":1e-6, "m":1e-3, "c":1e-2, "d":1e-1, "":1, "da":1e1, "h":1e2,
               "k":1e3, "M":1e6, "G":1e9, "T":1e12, "P":1e15, "E":1e18,
               "Z":1e21, "Y":1e24}

fundamental_units = ['metre', 'meter', 'gram', 'second', 'amp', 'kelvin', 'mole', 'candle']

#### DERIVED UNITS, from http://physics.nist.gov/cuu/Units/units.html
derived_unit_table = [
        [ 'radian',    'rad',  'get_or_create_dimension()' ],
        [ 'steradian', 'sr',   'get_or_create_dimension()' ],
        [ 'hertz',     'Hz',   'get_or_create_dimension(s= -1)' ],
        [ 'newton',    'N',    'get_or_create_dimension(m=1, kg=1, s=-2)' ],
        [ 'pascal',    'Pa',   'get_or_create_dimension(m= -1, kg=1, s=-2)' ],
        [ 'joule',     'J',    'get_or_create_dimension(m=2, kg=1, s=-2)' ],
        [ 'watt',      'W',    'get_or_create_dimension(m=2, kg=1, s=-3)' ],
        [ 'coulomb',   'C',    'get_or_create_dimension(s=1, A=1)' ],
        [ 'volt',      'V',    'get_or_create_dimension(m=2, kg=1, s=-3, A=-1)' ],
        [ 'farad',     'F',    'get_or_create_dimension(m= -2, kg=-1, s=4, A=2)' ],
        [ 'ohm',       'ohm',  'get_or_create_dimension(m=2, kg=1, s= -3, A=-2)' ],
        [ 'siemens',   'S',    'get_or_create_dimension(m= -2, kg=-1, s=3, A=2)' ],
        [ 'weber',     'Wb',   'get_or_create_dimension(m=2, kg=1, s=-2, A=-1)' ],
        [ 'tesla',     'T',    'get_or_create_dimension(kg=1, s=-2, A=-1)' ],
        [ 'henry',     'H',    'get_or_create_dimension(m=2, kg=1, s=-2, A=-2)' ],
        [ 'celsius',   'degC', 'get_or_create_dimension(K=1)' ],
        [ 'lumen',     'lm',   'get_or_create_dimension(cd=1)' ],
        [ 'lux',       'lx',   'get_or_create_dimension(m=-2, cd=1)' ],
        [ 'becquerel', 'Bq',   'get_or_create_dimension(s=-1)' ],
        [ 'gray',      'Gy',   'get_or_create_dimension(m=2, s=-2)' ],
        [ 'sievert',   'Sv',   'get_or_create_dimension(m=2, s=-2)' ],
        [ 'katal',     'kat',  'get_or_create_dimension(s=-1, mol=1)' ],
        ]

additional_units = '''
# Current list from http://physics.nist.gov/cuu/Units/units.html, far from complete
additional_units = [
    pascal * second, newton * metre, watt / metre ** 2, joule / kelvin,
    joule / (kilogram * kelvin), joule / kilogram, watt / (metre * kelvin),
    joule / metre ** 3, volt / metre ** 3, coulomb / metre ** 3, coulomb / metre ** 2,
    farad / metre, henry / metre, joule / mole, joule / (mole * kelvin),
    coulomb / kilogram, gray / second, katal / metre ** 3 ]
'''

## Generate derived unit objects and make a table of base units from these and the fundamental ones
base_units = fundamental_units+['gramme', 'kilogram']
derived = ''
for longname, shortname, definition in derived_unit_table:
    derived += '{longname} = Unit.create({definition}, "{longname}", "{shortname}")\n'.format(
                longname=longname, shortname=shortname, definition=definition)
    base_units.append(longname)

all_units = base_units + []

definitions = '######### SCALED BASE UNITS ###########\n'

# Generate scaled units for all base units
scaled_units = []
excluded_scaled_units = set()
for _bu in base_units:
    for _k in _siprefixes.keys():
        if len(_k):
            _u = _k+_bu
            all_units.append(_u)
            definitions += '{_u} = Unit.create_scaled_unit({_bu}, "{_k}")\n'.format(
                                                        _u=_u, _bu=_bu, _k=_k)
            if not _k in ["da", "d", "c", "h"]:
                scaled_units.append(_u)
            else:
                excluded_scaled_units.add(_u)

# Generate 2nd and 3rd powers for all scaled base units
definitions += '######### SCALED BASE UNITS TO POWERS ###########\n'
powered_units = []
for bu in all_units + []:
    for i in [2, 3]:
        u = bu+str(i)
        definitions += '{u} = {bu}**{i}\n'.format(u=u, bu=bu, i=i)
        definitions += '{u}.name = "{u}"\n'.format(u=u)
        all_units.append(u)
        if bu not in excluded_scaled_units:
            powered_units.append(u)

# Add unit names to __all__
all = '''
__all__ = [
{allunits}
    ]
'''.format(allunits='\n'.join('    "'+u+'",' for u in all_units))

def to_definition(name, x):
    return '''
{name} = [
{items}
    ]
'''.format(name=name, items='\n'.join('    '+i+',' for i in x))

template = open('units_template.py').read()
units_str = template.format(
    derived=derived, all=all, definitions=definitions,
    base_units=to_definition('base_units', base_units),
    scaled_units=to_definition('scaled_units', scaled_units),
    powered_units=to_definition('powered_units', powered_units),
    all_units=to_definition('all_units', all_units),
    additional_units=additional_units,
    )

open(units_fname, 'w').write(units_str)
