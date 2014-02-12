{% macro cpp_file() %}
#include<stdlib.h>
#include "objects.h"
#include<ctime>

{% for codeobj in code_objects %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

void brian_start()
{
	_init_arrays();
	_load_arrays();
	srand((unsigned int)time(NULL));
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}

{% for name, lines in run_funcs.items() %}
void {{name}}()
{
	{% for line in lines %}
	{{line}}
	{% endfor %}
}

{% endfor %}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}
#pragma once

void brian_start();
void brian_end();

{% for name, lines in run_funcs.items() %}
void {{name}}();
{% endfor %}

{% endmacro %}
