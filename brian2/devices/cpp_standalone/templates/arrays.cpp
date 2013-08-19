{% macro cpp_file() %}

#include "arrays.h"

{% for (varname, dtype_spec, N) in array_specs %}
{{dtype_spec}} *{{varname}} = new {{dtype_spec}}[{{N}}];
{% endfor %}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_ARRAYS_H
#define _BRIAN_ARRAYS_H

{% for (varname, dtype_spec, N) in array_specs %}
extern {{dtype_spec}} *{{varname}};
{% endfor %}

#endif

{% endmacro %}
