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
#include<map>
#include<tuple>
#include<cstdlib>
#include<string>

namespace brian {

std::string results_dir = "results/";  // can be overwritten by --results_dir command line arg
std::vector< rk_state* > _mersenne_twister_states;

//////////////// networks /////////////////
{% for net in networks | sort(attribute='name') %}
Network {{net.name}};
{% endfor %}

//////////////// array meta data //////////
std::map<std::string, std::tuple<bool, size_t, std::string, void*>> array_meta_data;

//////////////// set arrays by name ///////
void set_variable_by_name(std::string owner_variable, std::string s_value) {
	std::tuple<bool, size_t, std::string, void*> meta_data;
	try {
		meta_data = array_meta_data.at(owner_variable);
	} catch (const std::out_of_range& oor) {
    	std::cerr << "Did not find variable '" << owner_variable << "'" << std::endl;
  	}
	const bool is_dynamic = std::get<0>(meta_data);
	size_t var_size = std::get<1>(meta_data);  // will be overwritten by actual size for dynamic arrays
	size_t data_size;
	const std::string var_type = std::get<2>(meta_data);
	void* var_pointer = std::get<3>(meta_data);
	

	if (var_type == "double")
	{
		if (is_dynamic)
			var_size = ((std::vector<double>*)var_pointer)->size();
		data_size = var_size * sizeof(double);
	}
	else if (var_type == "float")
	{
		if (is_dynamic)
			var_size = ((std::vector<float>*)var_pointer)->size();
		data_size = var_size * sizeof(float);
	}
	else if (var_type == "int64_t")
	{
		if (is_dynamic)
			var_size = ((std::vector<int64_t>*)var_pointer)->size();
		data_size = var_size * sizeof(int64_t);
	}
	else if (var_type == "int32_t")
	{
		if (is_dynamic)
			var_size = ((std::vector<int32_t>*)var_pointer)->size();
		data_size = var_size * sizeof(int32_t);
	}

	if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9'))
	{
		#ifdef DEBUG
		std::cout << "Setting '" << owner_variable << "' to " << s_value << std::endl;
		#endif
		if (var_type == "double")
		{
			const double d_value = atof(s_value.c_str());
			for (size_t i = 0; i < var_size; i++)
			{
				((double *)var_pointer)[i] = d_value;
			}
		}
		else if (var_type == "float")
		{
			const float f_value = atof(s_value.c_str());
			for (size_t i = 0; i < var_size; i++)
				((float *)var_pointer)[i] = f_value;
		}
		else if (var_type == "int32_t")
		{
			const int32_t i32_value = atoi(s_value.c_str());
			for (size_t i = 0; i < var_size; i++)
				((int32_t *)var_pointer)[i] = i32_value;
		}
		else if (var_type == "int64_t")
		{
			const int64_t i64_value = atol(s_value.c_str());
			for (size_t i = 0; i < var_size; i++)
				((int64_t *)var_pointer)[i] = i64_value;
		}
	}
	else
	{ // file name
		ifstream f;
		#ifdef DEBUG
		std::cout << "Setting '" << owner_variable << "' from file '" << s_value << "'" << std::endl;
		#endif
		f.open(s_value, ios::in | ios::binary);
		if (f.is_open())
		{
			f.read(reinterpret_cast<char *>(var_pointer), data_size);
		}
		else
		{
			std::cerr << "Could not read '" << s_value << "'" << std::endl;
		}
	}
}
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

	array_meta_data = {
	{% for var, varname in array_specs | dictsort(by='value') %}
		{% if not var in dynamic_array_specs and not var in dynamic_array_2d_specs and not var.read_only %}
		{"{{var.owner.name}}.{{var.name}}", { false, {{var.size}}, "{{c_data_type(var.dtype)}}", {{varname}} } },
		{% endif %}
		{% if var in dynamic_array_specs and not var.read_only %}
		{# size will be directly requested from vector #}
		{"{{var.owner.name}}.{{var.name}}", { true, 0, "{{c_data_type(var.dtype)}}", {{varname}} } },
		{% endif %}
	{% endfor %}
	};

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
	outfile_{{varname}}.open(results_dir + "{{get_array_filename(var)}}", ios::binary | ios::out);
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
	outfile_{{varname}}.open(results_dir + "{{get_array_filename(var)}}", ios::binary | ios::out);
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
	outfile_{{varname}}.open(results_dir + "{{get_array_filename(var)}}", ios::binary | ios::out);
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
	outfile_profiling_info.open(results_dir + "profiling_info.txt", ios::out);
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
	// Write last run info to disk
	ofstream outfile_last_run_info;
	outfile_last_run_info.open(results_dir + "last_run_info.txt", ios::out);
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
#include "randomkit.h"
#include<vector>
{{ openmp_pragma('include') }}

namespace brian {

extern std::string results_dir;
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

void set_variable_by_name(std::string, std::string);

extern std::map<std::string, std::tuple<bool, size_t, std::string, void*>> array_meta_data;

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
