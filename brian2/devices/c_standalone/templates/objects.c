{# IS_OPENMP_COMPATIBLE #}
{% macro cpp_file() %}

#include<stdint.h>
#include "brianlib/clocks.h"
#include "objects.h"
#include "network.h"
#include<stdio.h>

{% for clock in clocks | sort(attribute='name') %}
Clock *{{clock.name}};
{% endfor %}

{% for net in networks | sort(attribute='name') %}
Network *{{net.name}};
{% endfor %}

//////////////// arrays ///////////////////
{% for var, varname in array_specs | dictsort(by='value') %}
{% if not var in dynamic_array_specs %}
{{c_data_type(var.dtype)}} * {{varname}};
const int _num_{{varname}} = {{var.size}};
{% endif %}
{% endfor %}

/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
{# arrays that are initialized from static data are already declared #}
{% if not name in array_specs.values() %}
{{dtype_spec}} * {{name}};
const int _num_{{name}} = {{N}};
{% endif %}
{% endfor %}

void _init_arrays()
{
    //////////////// clocks ///////////////////
    {% for clock in clocks | sort(attribute='name') %}
    {{clock.name}} = malloc(sizeof(Clock));
    Clock_construct({{clock.name}}, {{clock.dt_}}, 1e-14);
    {% endfor %}

    //////////////// networks /////////////////
    {% for net in networks | sort(attribute='name') %}
    {{net.name}} = malloc(sizeof(Network));
    Network_construct({{net.name}});
    {% endfor %}

    // Arrays initialized to 0
	{% for var in zero_arrays | sort(attribute='name') %}
	{% set varname = array_specs[var] %}
	{{varname}} = calloc(sizeof({{c_data_type(var.dtype)}}), {{var.size}});
    {% endfor %}

	// Arrays initialized to an "arange"
	{% for var, start in arange_arrays %}
	{% set varname = array_specs[var] %}
	{{varname}} = malloc(sizeof({{c_data_type(var.dtype)}}) * {{var.size}});
	{{ openmp_pragma('parallel-static') }}
	for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = {{start}} + i;
	{% endfor %}

	// static arrays
	{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
	{{name}} = malloc(sizeof({{dtype_spec}}) * {{N}});
	{% endfor %}
}

void _load_arrays()
{
	{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
	FILE* f{{name}};
	f{{name}} = fopen("static_arrays/{{name}}", "rb");
	fread({{name}}, sizeof({{dtype_spec}}), {{N}}, f{{name}});
    fclose(f{{name}});
	{% endfor %}
}	

void _write_arrays()
{
	{% for var, varname in array_specs | dictsort(by='value') %}
	{% if not (var in dynamic_array_specs or var in dynamic_array_2d_specs) %}
	FILE * outfile_{{varname}};

	outfile_{{varname}} = fopen("results/{{varname}}", "wb");
	fwrite({{varname}}, sizeof({{varname}}[0]), {{var.size}}, outfile_{{varname}});
	fclose(outfile_{{varname}});
    {% endif %}
	{% endfor %}
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
extern Network* magicnetwork;
{% for net in networks %}
extern Network* {{net.name}};
{% endfor %}


//////////////// arrays ///////////////////
{% for var, varname in array_specs | dictsort(by='value') %}
extern {{c_data_type(var.dtype)}} *{{varname}};
extern const int _num_{{varname}};
{% endfor %}


/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
{# arrays that are initialized from static data are already declared #}
{% if not name in array_specs.values() %}
extern {{dtype_spec}} *{{name}};
extern const int _num_{{name}};
{% endif %}
{% endfor %}

extern void _init_arrays();
extern void _load_arrays();
extern void _write_arrays();

#endif


{% endmacro %}
