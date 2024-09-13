{% macro cpp_file() %}

{% macro set_from_value(var_dtype, array_name) %}
{% if c_data_type(var_dtype) == 'double' %}
set_variable_from_value<double>(name, {{array_name}}, var_size, (double)atof(s_value.c_str()));
{% elif c_data_type(var_dtype) == 'float' %}
set_variable_from_value<float>(name, {{array_name}}, var_size, (float)atof(s_value.c_str()));
{% elif c_data_type(var_dtype) == 'int32_t' %}
set_variable_from_value<int32_t>(name, {{array_name}}, var_size, (int32_t)atoi(s_value.c_str()));
{% elif c_data_type(var_dtype) == 'int64_t' %}
set_variable_from_value<int64_t>(name, {{array_name}}, var_size, (int64_t)atol(s_value.c_str()));
{% elif c_data_type(var_dtype) == 'char' %}
set_variable_from_value(name, {{array_name}}, var_size, (char)atoi(s_value.c_str()));
{% endif %}
{%- endmacro %}

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<random>
#include<vector>
#include<iostream>
#include<fstream>
#include<map>
#include<tuple>
#include<cstdlib>
#include<string>

namespace brian {

std::string results_dir = "results/";  // can be overwritten by --results_dir command line arg

// For multhreading, we need one generator for each thread. We also create a distribution for
// each thread, even though this is not strictly necessary for the uniform distribution, as
// the distribution is stateless.
std::vector< RandomGenerator > _random_generators;

//////////////// networks /////////////////
{% for net in networks | sort(attribute='name') %}
Network {{net.name}};
{% endfor %}

void set_variable_from_value(std::string varname, char* var_pointer, size_t size, char value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << (value == 1 ? "True" : "False") << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_value(std::string varname, T* var_pointer, size_t size, T value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << value << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_file(std::string varname, T* var_pointer, size_t data_size, std::string filename) {
    ifstream f;
    streampos size;
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' from file '" << filename << "'" << std::endl;
    #endif
    f.open(filename, ios::in | ios::binary | ios::ate);
    size = f.tellg();
    if (size != data_size) {
        std::cerr << "Error reading '" << filename << "': file size " << size << " does not match expected size " << data_size << std::endl;
        return;
    }
    f.seekg(0, ios::beg);
    if (f.is_open())
        f.read(reinterpret_cast<char *>(var_pointer), data_size);
    else
        std::cerr << "Could not read '" << filename << "'" << std::endl;
    if (f.fail())
        std::cerr << "Error reading '" << filename << "'" << std::endl;
}

//////////////// set arrays by name ///////
void set_variable_by_name(std::string name, std::string s_value) {
    size_t var_size;
    size_t data_size;
    // C-style or Python-style capitalization is allowed for boolean values
    if (s_value == "true" || s_value == "True")
        s_value = "1";
    else if (s_value == "false" || s_value == "False")
        s_value = "0";
    // non-dynamic arrays
    {% for var, varname in array_specs | dictsort(by='value') %}
    {% if not var in dynamic_array_specs and not var.read_only %}
    if (name == "{{var.owner.name}}.{{var.name}}") {
        var_size = {{var.size}};
        data_size = {{var.size}}*sizeof({{c_data_type(var.dtype)}});
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            {{ set_from_value(var.dtype, get_array_name(var)) }}
        } else {
            // set from file
            set_variable_from_file(name, {{get_array_name(var)}}, data_size, s_value);
        }
        return;
    }
    {% endif %}
    {% endfor %}
    // dynamic arrays (1d)
    {% for var, varname in dynamic_array_specs | dictsort(by='value') %}
    {% if not var.read_only %}
    if (name == "{{var.owner.name}}.{{var.name}}") {
        var_size = {{get_array_name(var, access_data=False)}}.size();
        data_size = var_size*sizeof({{c_data_type(var.dtype)}});
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            {{ set_from_value(var.dtype, "&" + get_array_name(var, False) + "[0]") }}
        } else {
            // set from file
            set_variable_from_file(name, &{{get_array_name(var, False)}}[0], data_size, s_value);
        }
        return;
    }
    {% endif %}
    {% endfor %}
    {% for var, varname in timed_arrays | dictsort(by='value') %}
    if (name == "{{varname}}.values") {
        var_size = {{var.values.size}};
        data_size = var_size*sizeof({{c_data_type(var.values.dtype)}});
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            {{ set_from_value(var.values.dtype, varname + "_values") }}

        } else {
            // set from file
            set_variable_from_file(name, {{varname}}_values, data_size, s_value);
        }
        return;
    }
    {% endfor %}
    std::cerr << "Cannot set unknown variable '" << name << "'." << std::endl;
    exit(1);
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

    // Random number generator states
    std::random_device rd;
    for (int i=0; i<{{openmp_pragma('get_num_threads')}}; i++)
        _random_generators.push_back(RandomGenerator());
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
#include<random>
#include<vector>
{{ openmp_pragma('include') }}

namespace brian {

extern std::string results_dir;

class RandomGenerator {
    private:
        std::mt19937 gen;
        double stored_gauss;
        bool has_stored_gauss = false;
    public:
        RandomGenerator() {
            seed();
        }
        void seed() {
            std::random_device rd;
            gen.seed(rd());
            has_stored_gauss = false;
        }
        void seed(unsigned long seed) {
            gen.seed(seed);
            has_stored_gauss = false;
        }
        double rand() {
            /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
            const long a = gen() >> 5;
            const long b = gen() >> 6;
            return (a * 67108864.0 + b) / 9007199254740992.0;
        }

        double randn() {
            if (has_stored_gauss) {
                const double tmp = stored_gauss;
                has_stored_gauss = false;
                return tmp;
            }
            else {
                double f, x1, x2, r2;

                do {
                    x1 = 2.0*rand() - 1.0;
                    x2 = 2.0*rand() - 1.0;
                    r2 = x1*x1 + x2*x2;
                }
                while (r2 >= 1.0 || r2 == 0.0);

                /* Box-Muller transform */
                f = sqrt(-2.0*log(r2)/r2);
                /* Keep for next call */
                stored_gauss = f*x1;
                has_stored_gauss = true;
                return f*x2;
            }
        }
};

// In OpenMP we need one state per thread
extern std::vector< RandomGenerator > _random_generators;

//////////////// clocks ///////////////////
{% for clock in clocks | sort(attribute='name') %}
extern Clock {{clock.name}};
{% endfor %}

//////////////// networks /////////////////
{% for net in networks | sort(attribute='name') %}
extern Network {{net.name}};
{% endfor %}



void set_variable_by_name(std::string, std::string);

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
