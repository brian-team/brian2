{% import 'common_macros.cpp' as common with context %}

{% macro main() %}
	{{ common.insert_lines_commented('SUPPORT CODE', support_code_lines) }}
	{{ common.insert_lines('HANDLE DENORMALS', denormals_code_lines) }}
	{{ common.insert_lines('HASH DEFINES', hashdefine_lines) }}
	{{ common.insert_lines('POINTERS', pointers_lines) }}
	//// MAIN CODE ////////////
	for(int _idx=0; _idx<N; _idx++)
	{
	    const int _vectorisation_idx = _idx;
	    {{ common.insert_lines('CONDITION', code_lines['condition']) }}
		if(_cond) {
			{{ common.insert_lines('STATEMENT', code_lines['statement']) }}
		}
	}
{% endmacro %}

{% macro support_code() %}
	{{ common.insert_lines('SUPPORT CODE', support_code_lines) }}
{% endmacro %}
