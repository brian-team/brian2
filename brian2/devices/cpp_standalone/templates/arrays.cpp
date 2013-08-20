{% macro cpp_file() %}

#include<stdint.h>
#include "arrays.h"

{% for (varname, dtype_spec, N) in array_specs %}
{{dtype_spec}} *{{varname}} = new {{dtype_spec}}[{{N}}];
{% endfor %}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_ARRAYS_H
#define _BRIAN_ARRAYS_H

#include<stdint.h>

{% for (varname, dtype_spec, N) in array_specs %}
extern {{dtype_spec}} *{{varname}};
{% endfor %}

#endif

{% endmacro %}
