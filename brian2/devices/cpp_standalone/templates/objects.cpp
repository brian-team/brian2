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

// Helper function to get file size
// Adapted from https://www.joelverhagen.com/blog/2011/03/get-the-size-of-a-file-in-c/
int get_file_size(const std::string &filename)
{
    ifstream file(filename.c_str(), ifstream::in | ifstream::binary);

    if(!file.is_open())
    {
        return -1;
    }

    file.seekg(0, ios::end);
    int filesize = file.tellg();
    file.close();

    return filesize;
}

void brian::_init_arrays()
{
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

    // static arrays are allocated in _load_arrays because we need their sizes

    // Random number generator states
    for (int i=0; i<{{openmp_pragma('get_num_threads')}}; i++)
        _mersenne_twister_states.push_back(new rk_state());
}

void brian::_load_arrays()
{
    {% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
    // infer array size from disk array size
    _num_{{name}} = get_file_size("static_arrays/{{name}}")/sizeof({{dtype_spec}});
    // allocate memory
    {% if name in dynamic_array_specs.values() %}
    {{name}}.resize(_num_{{name}});
    {% else %}
    {{name}} = new {{dtype_spec}}[_num_{{name}}];
    {% endif %}
    // load data
    ifstream f{{name}};
    f{{name}}.open("static_arrays/{{name}}", ios::in | ios::binary);
    if(f{{name}}.is_open())
    {
        {% if name in dynamic_array_specs.values() %}
        f{{name}}.read(reinterpret_cast<char*>(&{{name}}[0]), _num_{{name}}*sizeof({{dtype_spec}}));
        {% else %}
        f{{name}}.read(reinterpret_cast<char*>({{name}}), _num_{{name}}*sizeof({{dtype_spec}}));
        {% endif %}
    } else
    {
        std::cout << "Error opening static array {{name}}." << endl;
    }
    {% endfor %}
}

void brian::_write_arrays()
{
    {% for var, varname in array_specs | dictsort(by='value') %}
    {% if not (var in dynamic_array_specs or var in dynamic_array_2d_specs) %}
    {% if not vars_to_write or var in vars_to_write %}
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
    {% endif %}
    {% endfor %}

    {% for var, varname in dynamic_array_specs | dictsort(by='value') %}
    {% if not vars_to_write or var in vars_to_write %}
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
    {% endif %}
    {% endfor %}

    {% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
    {% if not vars_to_write or var in vars_to_write %}
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
    {% endif %}
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

void brian::_dealloc_arrays()
{
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

class brian {
public:

    std::vector< rk_state* > _mersenne_twister_states;

    //////////////// networks /////////////////
    {% for net in networks | sort(attribute='name') %}
    Network {{net.name}};
    {% endfor %}

    //////////////// arrays ///////////////////
    {% for var, varname in array_specs | dictsort(by='value') %}
    {% if not var in dynamic_array_specs %}
    {{c_data_type(var.dtype)}} * {{varname}};
    static const int _num_{{varname}} = {{var.size}};
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
    int _num_{{name}};
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

    {% if profiled_codeobjects is defined %}
    // Profiling information for each code object
    {% for codeobj in profiled_codeobjects | sort %}
    double {{codeobj}}_profiling_info = 0.0;
    {% endfor %}
    {% endif %}

    ////////////////// parameters //////////////
    void _set_default_parameters();
    int _read_command_line_parameters(int argc, char *argv[]);
    {% for name, param in parameters | dictsort(by='key') %}
    {{c_data_type(param.dtype)}} _parameter_{{name}};
    {% endfor %}

    //////////////// basic methods /////////////
    void _init_arrays();
    void _load_arrays();
    void _write_arrays();
    void _dealloc_arrays();
    void _start();
    void _end();

    ///////////// code object methods //////////
    {% for codeobj in code_objects | sort(attribute='name') %}
    void _run_{{codeobj.name}}();
    void _debugmsg_{{codeobj.name}}();
    {% endfor %}

    //////////// run functions /////////////////
    {% for name, lines in run_funcs.items() | sort(attribute='name') %}
    void _run_func_{{name}}();
    {% endfor %}

    ////////// main /////////////////////////////
    void _run_main_lines();
};

#endif

{% endmacro %}
