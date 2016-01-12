{% macro cpp_file() %}

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<vector>
#include<iostream>
#include<fstream>

namespace brian {

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
SynapticPathway<double> {{path.name}}(
		{{dynamic_array_specs[path.variables['delay']]}},
		{{dynamic_array_specs[path.synapse_sources]}},
		{{path.source.start}}, {{path.source.stop}});
{% endfor %}
{% endfor %}

//////////////// clocks ///////////////////
{% for clock in clocks | sort(attribute='name') %}
Clock {{clock.name}};  // attributes will be set in run.cpp
{% endfor %}

// Profiling information for each code object
{% for codeobj in code_objects | sort(attribute='name') %}
double {{codeobj.name}}_profiling_info = 0.0;
{% endfor %}

}

void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	{% for var in zero_arrays | sort(attribute='name') %}
	{% if var in dynamic_array_specs %}
	{% set varname = '_dynamic'+array_specs[var] %}
	{% else %}
	{% set varname = array_specs[var] %}
	{% endif %}
	{% if varname in dynamic_array_specs.values() %}
	{{varname}}.resize({{var.size}});
	{% else %}
	{{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
	{% endif %}
	{{ openmp_pragma('parallel-static') }}
	for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = 0;
	{% endfor %}

	// Arrays initialized to an "arange"
	{% for var, start in arange_arrays %}
	{% set varname = array_specs[var] %}
	{% if varname in dynamic_array_specs.values() %}
	{{varname}}.resize({{var.size}});
	{% else %}
	{{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
	{% endif %}
	{{ openmp_pragma('parallel-static') }}
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
            {% if var.is_boolean %}
            // Copy the boolean vector to a char vector so we don't have to deal with its internal representation
            std::vector<char> _arr_copy = std::vector<char>({{varname}}.size());
            std::copy({{varname}}.begin(), {{varname}}.end(), _arr_copy.begin());
            outfile_{{varname}}.write(reinterpret_cast<char*>(&_arr_copy[0]), _arr_copy.size()*sizeof(_arr_copy[0]));
            {% else %}
			outfile_{{varname}}.write(reinterpret_cast<char*>(&{{varname}}[0]), {{varname}}.size()*sizeof({{varname}}[0]));
			{% endif %}
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
            {% if var.is_boolean %}
                // Copy the boolean vector to a char vector so we don't have to deal with its internal representation
                std::vector<char> _arr_copy = std::vector<char>({{varname}}.m);
                std::copy({{varname}}(n).begin(), {{varname}}(n).end(), _arr_copy.begin());
                outfile_{{varname}}.write(reinterpret_cast<char*>(&_arr_copy[0]), _arr_copy.size()*sizeof(_arr_copy[0]));
            {% else %}
                outfile_{{varname}}.write(reinterpret_cast<char*>(&{{varname}}(n, 0)), {{varname}}.m*sizeof({{varname}}(0, 0)));
            {% endif %}
            }
        }
        outfile_{{varname}}.close();
	} else
	{
		std::cout << "Error writing output file for {{varname}}." << endl;
	}
	{% endfor %}

	// Write profiling info to disk
	ofstream outfile_profiling_info;
	outfile_profiling_info.open("results/profiling_info.txt", ios::out);
	if(outfile_profiling_info.is_open())
	{
	{% for codeobj in code_objects | sort(attribute='name') %}
	outfile_profiling_info << "{{codeobj.name}}\t" << {{codeobj.name}}_profiling_info << std::endl;
	{% endfor %}
	outfile_profiling_info.close();
	} else
	{
	    std::cout << "Error writing profiling info to file." << std::endl;
	}

	// Write last run info to disk
	ofstream outfile_last_run_info;
	outfile_last_run_info.open("results/last_run_info.txt", ios::out);
	if(outfile_last_run_info.is_open())
	{
		outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
		outfile_last_run_info.close();
	} else
	{
	    std::cout << "Error writing last run info to file." << std::endl;
	}
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
#include<vector>
{{ openmp_pragma('include') }}

namespace brian {

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
{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
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
extern SynapticPathway<double> {{path.name}};
{% endfor %}
{% endfor %}

// Profiling information for each code object
{% for codeobj in code_objects | sort(attribute='name') %}
extern double {{codeobj.name}}_profiling_info;
{% endfor %}

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


{% endmacro %}
