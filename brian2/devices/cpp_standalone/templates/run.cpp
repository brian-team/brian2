{# IS_OPENMP_COMPATIBLE #}
{% macro cpp_file() %}
#include<stdlib.h>
#include "objects.h"
#include<ctime>

{% for codeobj in code_objects | sort(attribute='name') %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

{% for name in user_headers %}
#include {{name}}
{% endfor %}

void brian_start()
{
	_init_arrays();
	_load_arrays();
	srand((unsigned int)time(NULL));
	rand(); // put this in because the first random number generated on some versions of C++ is always almost the same
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}

{% for name, lines in run_funcs.items() | sort(attribute='name') %}
void {{name}}()
{
	using namespace brian;

    {{lines|autoindent}}
}

{% endfor %}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

void brian_start();
void brian_end();

{% for name, lines in run_funcs.items() | sort(attribute='name') %}
void {{name}}();
{% endfor %}

{% endmacro %}
