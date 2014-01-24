{% macro cpp_file() %}

#include<stdint.h>
#include<vector>
#include "objects.h"
#include "brianlib/synapses.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/network.h"
#include<iostream>
#include<fstream>

//////////////// clocks ///////////////////
{% for clock in clocks %}
Clock {{clock.name}}({{clock.dt_}});
{% endfor %}

//////////////// networks /////////////////
{% for net in networks %}
Network {{net.name}};
{% endfor %}

//////////////// arrays ///////////////////
{% for var, varname in array_specs.items() %}
{% if not var in dynamic_array_specs %}
{{c_data_type(var.dtype)}} *{{varname}};
const int _num_{{varname}} = {{var.size}};
{% endif %}
{% endfor %}

//////////////// dynamic arrays 1d /////////
{% for var, varname in dynamic_array_specs.items() %}
std::vector<{{c_data_type(var.dtype)}}> {{varname}};
{% endfor %}

//////////////// dynamic arrays 2d /////////
{% for var, varname in dynamic_array_2d_specs.items() %}
DynamicArray2D<{{c_data_type(var.dtype)}}> {{varname}};
{% endfor %}

/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs %}
{{dtype_spec}} *{{name}};
const int _num_{{name}} = {{N}};
{% endfor %}

//////////////// synapses /////////////////
{% for S in synapses %}
// {{S.name}}
Synapses<double> {{S.name}}({{S.source|length}}, {{S.target|length}});
{% for path in S._pathways %}
SynapticPathway<double> {{path.name}}(
		{{path.source|length}}, {{path.target|length}},
		{{dynamic_array_specs[path.variables['delay']]}},
		{{dynamic_array_specs[path.synapse_sources]}},
		{{path.source.dt_}},
		{{path.source.start}}, {{path.source.stop}}
		);
{% endfor %}
{% endfor %}


void _init_arrays()
{
    // Arrays initialized to 0
	{% for var in zero_arrays %}
	{% set varname = array_specs[var] %}
	{{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
	for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = 0;
	{% endfor %}

	// Arrays initialized to an "arange"
	{% for var, start in arange_arrays %}
	{% set varname = array_specs[var] %}
	{{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
	for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = {{start}} + i;
	{% endfor %}

	// static arrays
	{% for (name, dtype_spec, N, filename) in static_array_specs %}
	{{name}} = new {{dtype_spec}}[{{N}}];
	{% endfor %}
}

void _load_arrays()
{
	{% for (name, dtype_spec, N, filename) in static_array_specs %}
	ifstream f{{name}};
	f{{name}}.open("static_arrays/{{name}}", ios::in | ios::binary);
	if(f{{name}}.is_open())
	{
		f{{name}}.read(reinterpret_cast<char*>({{name}}), {{N}}*sizeof({{dtype_spec}}));
	} else
	{
		cout << "Error opening static array {{name}}." << endl;
	}
	{% endfor %}
}

void _write_arrays()
{
	{% for var, varname in array_specs.items() %}
	{% if not var in dynamic_array_specs %}
	ofstream outfile_{{varname}};
	outfile_{{varname}}.open("results/{{varname}}", ios::binary | ios::out);
	if(outfile_{{varname}}.is_open())
	{
		outfile_{{varname}}.write(reinterpret_cast<char*>({{varname}}), {{var.size}}*sizeof({{varname}}[0]));
		outfile_{{varname}}.close();
	} else
	{
		cout << "Error writing output file for {{varname}}." << endl;
	}
	{% endif %}
	{% endfor %}

	{% for var, varname in dynamic_array_specs.items() %}
	ofstream outfile_{{varname}};
	outfile_{{varname}}.open("results/{{varname}}", ios::binary | ios::out);
	if(outfile_{{varname}}.is_open())
	{
		outfile_{{varname}}.write(reinterpret_cast<char*>(&{{varname}}[0]), {{varname}}.size()*sizeof({{varname}}[0]));
		outfile_{{varname}}.close();
	} else
	{
		cout << "Error writing output file for {{varname}}." << endl;
	}
	{% endfor %}
}

void _dealloc_arrays()
{
	{% for var, varname in array_specs.items() %}
	{% if not var in dynamic_array_specs %}
	if({{varname}}!=0)
	{
		delete [] {{varname}};
		{{varname}} = 0;
	}
	{% endif %}
	{% endfor %}

	// static arrays
	{% for (name, dtype_spec, N, filename) in static_array_specs %}
	if({{name}}!=0)
	{
		delete [] {{name}};
		{{name}} = 0;
	}
	{% endfor %}
}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include<vector>
#include<stdint.h>
#include "brianlib/synapses.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/network.h"

//////////////// clocks ///////////////////
{% for clock in clocks %}
extern Clock {{clock.name}};
{% endfor %}

//////////////// networks /////////////////
extern Network magicnetwork;
{% for net in networks %}
extern Network {{net.name}};
{% endfor %}


//////////////// dynamic arrays ///////////
{% for var, varname in dynamic_array_specs.items() %}
extern std::vector<{{c_data_type(var.dtype)}}> {{varname}};
{% endfor %}

//////////////// arrays ///////////////////
{% for var, varname in array_specs.items() %}
{% if not var in dynamic_array_specs %}
extern {{c_data_type(var.dtype)}} *{{varname}};
extern const int _num_{{varname}};
{% endif %}
{% endfor %}

//////////////// dynamic arrays 2d /////////
{% for var, varname in dynamic_array_2d_specs.items() %}
extern DynamicArray2D<{{c_data_type(var.dtype)}}> {{varname}};
{% endfor %}

/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs %}
extern {{dtype_spec}} *{{name}};
extern const int _num_{{name}};
{% endfor %}

//////////////// synapses /////////////////
{% for S in synapses %}
// {{S.name}}
extern Synapses<double> {{S.name}};
{% for path in S._pathways %}
extern SynapticPathway<double> {{path.name}};
{% endfor %}
{% endfor %}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


{% endmacro %}
