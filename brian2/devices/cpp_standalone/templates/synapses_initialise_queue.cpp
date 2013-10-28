{% macro cpp_file() %}
#include "code_objects/{{codeobj_name}}.h"
void _run_{{codeobj_name}}(double t) {}
{% endmacro %}

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

void _run_{{codeobj_name}}(double t);

#endif
{% endmacro %}
