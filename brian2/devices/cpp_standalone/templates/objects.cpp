{% macro cpp_file() %}

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>
#include<iostream>
#include<fstream>

namespace brian {

std::vector< rk_state* > _mersenne_twister_states;

//////////////// networks /////////////////
{% for net in networks | sort(attribute='name') %}
Network {{net.name}};
{% endfor %}

//////////////// arrays ///////////////////
{% for var, varname in array_specs | dictsort(by='value') %}
{% if not var in dynamic_array_specs %}
{{c_data_type(var.dtype)}} * {{varname}};
const int _num_{{varname}} = {{var.size}};
{% endif %}
{% endfor %}

//////////////// dynamic arrays 1d /////////
{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
std::vector<{{c_data_type(var.dtype)}}> {{varname}};
{% endfor %}

//////////////// dynamic arrays 2d /////////
{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
DynamicArray2D<{{c_data_type(var.dtype)}}> {{varname}};
{% endfor %}

/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
{# arrays that are initialized from static data are already declared #}
{% if not (name in array_specs.values() or name in dynamic_array_specs.values() or name in dynamic_array_2d_specs.values())%}
{{dtype_spec}} * {{name}};
const int _num_{{name}} = {{N}};
{% endif %}
{% endfor %}

//////////////// synapses /////////////////
{% for S in synapses | sort(attribute='name') %}
// {{S.name}}
{% for path in S._pathways | sort(attribute='name') %}
SynapticPathway {{path.name}}(
		{{dynamic_array_specs[path.synapse_sources]}},
		{{path.source.start}}, {{path.source.stop}});
{% endfor %}
{% endfor %}

//////////////// clocks ///////////////////
{% for clock in clocks | sort(attribute='name') %}
Clock {{clock.name}};  // attributes will be set in run.cpp
{% endfor %}

{% if profiled_codeobjects is defined %}
// Profiling information for each code object
{% for codeobj in profiled_codeobjects | sort %}
double {{codeobj}}_profiling_info = 0.0;
{% endfor %}
{% endif %}
}

void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	{% for var, varname in zero_arrays | sort(attribute='1') %}
	{% if varname in dynamic_array_specs.values() %}
	{{varname}}.resize({{var.size}});
	{% else %}
	{{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
	{% endif %}
    {{ openmp_pragma('parallel-static')}}
	for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = 0;

	{% endfor %}

	// Arrays initialized to an "arange"
	{% for var, varname, start in arange_arrays | sort(attribute='1')%}
	{% if varname in dynamic_array_specs.values() %}
	{{varname}}.resize({{var.size}});
	{% else %}
	{{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
	{% endif %}
    {{ openmp_pragma('parallel-static')}}
	for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = {{start}} + i;

	{% endfor %}

	// static arrays
	{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
	{% if name in dynamic_array_specs.values() %}
	{{name}}.resize({{N}});
	{% else %}
	{{name}} = new {{dtype_spec}}[{{N}}];
	{% endif %}
	{% endfor %}

	// Random number generator states
	for (int i=0; i<{{openmp_pragma('get_num_threads')}}; i++)
	    _mersenne_twister_states.push_back(new rk_state());
}

void _load_arrays()
{
	using namespace brian;

	{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
	ifstream f{{name}};
	f{{name}}.open("static_arrays/{{name}}", ios::in | ios::binary);
	if(f{{name}}.is_open())
	{
	    {% if name in dynamic_array_specs.values() %}
	    f{{name}}.read(reinterpret_cast<char*>(&{{name}}[0]), {{N}}*sizeof({{dtype_spec}}));
	    {% else %}
		f{{name}}.read(reinterpret_cast<char*>({{name}}), {{N}}*sizeof({{dtype_spec}}));
		{% endif %}
	} else
	{
		std::cout << "Error opening static array {{name}}." << endl;
	}
	{% endfor %}
}

void _write_arrays()
{
	using namespace brian;

	{% for var, varname in array_specs | dictsort(by='value') %}
	{% if not (var in dynamic_array_specs or var in dynamic_array_2d_specs) %}
	ofstream outfile_{{varname}};
	outfile_{{varname}}.open("{{get_array_filename(var) | replace('\\', '\\\\')}}", ios::binary | ios::out);
	if(outfile_{{varname}}.is_open())
	{
		outfile_{{varname}}.write(reinterpret_cast<char*>({{varname}}), {{var.size}}*sizeof({{get_array_name(var)}}[0]));
		outfile_{{varname}}.close();
	} else
	{
		std::cout << "Error writing output file for {{varname}}." << endl;
	}
	{% endif %}
	{% endfor %}

	{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
	ofstream outfile_{{varname}};
	outfile_{{varname}}.open("{{get_array_filename(var) | replace('\\', '\\\\')}}", ios::binary | ios::out);
	if(outfile_{{varname}}.is_open())
	{
        if (! {{varname}}.empty() )
        {
			outfile_{{varname}}.write(reinterpret_cast<char*>(&{{varname}}[0]), {{varname}}.size()*sizeof({{varname}}[0]));
		    outfile_{{varname}}.close();
		}
	} else
	{
		std::cout << "Error writing output file for {{varname}}." << endl;
	}
	{% endfor %}

	{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
	ofstream outfile_{{varname}};
	outfile_{{varname}}.open("{{get_array_filename(var) | replace('\\', '\\\\')}}", ios::binary | ios::out);
	if(outfile_{{varname}}.is_open())
	{
        for (int n=0; n<{{varname}}.n; n++)
        {
            if (! {{varname}}(n).empty())
            {
                outfile_{{varname}}.write(reinterpret_cast<char*>(&{{varname}}(n, 0)), {{varname}}.m*sizeof({{varname}}(0, 0)));
            }
        }
        outfile_{{varname}}.close();
	} else
	{
		std::cout << "Error writing output file for {{varname}}." << endl;
	}
	{% endfor %}
    {% if profiled_codeobjects is defined and profiled_codeobjects %}
	// Write profiling info to disk
	ofstream outfile_profiling_info;
	outfile_profiling_info.open("results/profiling_info.txt", ios::out);
	if(outfile_profiling_info.is_open())
	{
	{% for codeobj in profiled_codeobjects | sort %}
	outfile_profiling_info << "{{codeobj}}\t" << {{codeobj}}_profiling_info << std::endl;
	{% endfor %}
	outfile_profiling_info.close();
	} else
	{
	    std::cout << "Error writing profiling info to file." << std::endl;
	}
    {% endif %}
}

void _dealloc_arrays()
{
	using namespace brian;

	{% for var, varname in array_specs | dictsort(by='value') %}
	{% if varname in dynamic_array_specs.values() %}
	if({{varname}}!=0)
	{
		delete [] {{varname}};
		{{varname}} = 0;
	}
	{% endif %}
	{% endfor %}

	// static arrays
	{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
	{% if not name in dynamic_array_specs.values() %}
	if({{name}}!=0)
	{
		delete [] {{name}};
		{{name}} = 0;
	}
	{% endif %}
	{% endfor %}
}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>
{{ openmp_pragma('include') }}

namespace brian {

// In OpenMP we need one state per thread
extern std::vector< rk_state* > _mersenne_twister_states;

//////////////// clocks ///////////////////
{% for clock in clocks | sort(attribute='name') %}
extern Clock {{clock.name}};
{% endfor %}

//////////////// networks /////////////////
{% for net in networks | sort(attribute='name') %}
extern Network {{net.name}};
{% endfor %}

//////////////// dynamic arrays ///////////
{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
extern std::vector<{{c_data_type(var.dtype)}}> {{varname}};
{% endfor %}

//////////////// arrays ///////////////////
{% for var, varname in array_specs | dictsort(by='value') %}
{% if not var in dynamic_array_specs %}
extern {{c_data_type(var.dtype)}} *{{varname}};
extern const int _num_{{varname}};
{% endif %}
{% endfor %}

//////////////// dynamic arrays 2d /////////
{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
extern DynamicArray2D<{{c_data_type(var.dtype)}}> {{varname}};
{% endfor %}

/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs | sort(attribute='0') %}
{# arrays that are initialized from static data are already declared #}
{% if not (name in array_specs.values() or name in dynamic_array_specs.values() or name in dynamic_array_2d_specs.values())%}
extern {{dtype_spec}} *{{name}};
extern const int _num_{{name}};
{% endif %}
{% endfor %}

//////////////// synapses /////////////////
{% for S in synapses | sort(attribute='name') %}
// {{S.name}}
{% for path in S._pathways | sort(attribute='name') %}
extern SynapticPathway {{path.name}};
{% endfor %}
{% endfor %}

{% if profiled_codeobjects is defined %}
// Profiling information for each code object
{% for codeobj in profiled_codeobjects | sort %}
extern double {{codeobj}}_profiling_info;
{% endfor %}
{% endif %}
}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


{% endmacro %}
