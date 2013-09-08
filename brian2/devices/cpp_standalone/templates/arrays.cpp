{% macro cpp_file() %}

#include<stdint.h>
#include<vector>
#include "arrays.h"

// static arrays
{% for (varname, dtype_spec, N) in array_specs %}
{{dtype_spec}} *{{varname}} = new {{dtype_spec}}[{{N}}];
{% endfor %}

// dynamic arrays
{% for (varname, dtype_spec) in dynamic_array_specs %}
std::vector<{{dtype_spec}}> {{varname}};
{% endfor %}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_ARRAYS_H
#define _BRIAN_ARRAYS_H

#include<vector>
#include<stdint.h>

// static arrays
{% for (varname, dtype_spec, N) in array_specs %}
extern {{dtype_spec}} *{{varname}};
{% endfor %}

// dynamic arrays
{% for (varname, dtype_spec) in dynamic_array_specs %}
extern std::vector<{{dtype_spec}}> {{varname}};
{% endfor %}

#endif

{% endmacro %}
