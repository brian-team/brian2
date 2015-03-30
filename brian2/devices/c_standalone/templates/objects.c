{# IS_OPENMP_COMPATIBLE #}
{% macro cpp_file() %}

#include<stdint.h>
#include "brianlib/clocks.h"
#include "objects.h"
#include "network.h"
#include<stdio.h>

{% for clock in clocks | sort(attribute='name') %}
Clock _{{clock.name}};
Clock *{{clock.name}} = &_{{clock.name}};
{% endfor %}

{% for net in networks | sort(attribute='name') %}
Network _{{net.name}};
Network *{{net.name}} = &_{{net.name}};
{% endfor %}

//////////////// arrays ///////////////////
{% for var, varname in array_specs | dictsort(by='value') %}
{% if (not var in dynamic_array_specs) and (not varname in static_arrays) %}
{{c_data_type(var.dtype)}} {{varname}}[{{var.length}}] = {0,};
const int _num_{{varname}} = {{var.length}};
{% endif %}
{% endfor %}

/////////////// static arrays /////////////
{% for name, var in static_arrays | dictsort(by='key') %}
{% set N = var.size %}
{{c_data_type(var.dtype)}} {{name}}[{{N}}] = { {% for val in var %} {{val}}, {% endfor %} };
const int _num_{{name}} = {{N}};
{% endfor %}

void _init_arrays()
{
    //////////////// clocks ///////////////////
    {% for clock in clocks | sort(attribute='name') %}
    Clock_construct({{clock.name}}, {{clock.dt_}}, 1e-14);
    {% endfor %}

	// Arrays initialized to an "arange"
	{% for var, start in arange_arrays %}
	{% set varname = array_specs[var] %}
	{{ openmp_pragma('parallel-static') }}
	for(int i=0; i<{{var.length}}; i++) {{varname}}[i] = {{start}} + i;
	{% endfor %}
}

void _write_arrays()
{
    {% if write_arrays %}
	{% for var, varname in array_specs | dictsort(by='value') %}
	{% if not (var in dynamic_array_specs or var in dynamic_array_2d_specs) %}
	FILE * outfile_{{varname}};

	outfile_{{varname}} = fopen("results/{{varname}}", "wb");
	fwrite({{varname}}, sizeof({{varname}}[0]), {{var.length}}, outfile_{{varname}});
	fclose(outfile_{{varname}});
    {% endif %}
	{% endfor %}
	{% else %}
	// Writing arrays to disk has been disabled
	{% endif %}
}


{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include<stdint.h>
#include "brianlib/clocks.h"
#include "network.h"
{{ openmp_pragma('include') }}


//////////////// clocks ///////////////////
{% for clock in clocks %}
extern Clock* {{clock.name}};
{% endfor %}

//////////////// networks /////////////////
{% for net in networks %}
extern Network* {{net.name}};
{% endfor %}


//////////////// arrays ///////////////////
{% for var, varname in array_specs | dictsort(by='value') %}
extern {{c_data_type(var.dtype)}} {{varname}}[];
extern const int _num_{{varname}};
{% endfor %}


/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
{# arrays that are initialized from static data are already declared #}
{% if not name in array_specs.values() %}
extern {{dtype_spec}} {{name}}[];
extern const int _num_{{name}};
{% endif %}
{% endfor %}

extern void _init_arrays();
extern void _write_arrays();

#endif


{% endmacro %}
