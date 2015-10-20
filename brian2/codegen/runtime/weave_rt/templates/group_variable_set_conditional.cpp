{% import 'common_macros.cpp' as common with context %}
{# USES_VARIABLES { N } #}
{# ALLOWS_SCALAR_WRITE #}

{% macro main() %}
	{{ common.insert_lines_commented('SUPPORT CODE', support_code_lines) }}
	{{ common.insert_lines('HANDLE DENORMALS', denormals_code_lines) }}
	{{ common.insert_lines('HASH DEFINES', hashdefine_lines) }}
	{{ common.insert_lines('POINTERS', pointers_lines) }}
	//// MAIN CODE ////////////
    // This allows everything to work correctly for synapses where N is not a
    // constant
    const int _N = {{constant_or_scalar('N', variables['N'])}};
	// scalar code
	const int _vectorisation_idx = 1;
	{# Note that the scalar_code['statement'] will not write to any scalar
	   variables (except if the condition is simply 'True' and no vector code
	   is present), it will only read in scalar variables that are used by the
	   vector code. #}
	{{scalar_code['condition']|autoindent}}
	{{scalar_code['statement']|autoindent}}

	for(int _idx=0; _idx<_N; _idx++)
	{
	    const int _vectorisation_idx = _idx;
	    {{ common.insert_lines('CONDITION', vector_code['condition']) }}
		if(_cond) {
			{{ common.insert_lines('STATEMENT', vector_code['statement']) }}
		}
	}
{% endmacro %}

{% macro support_code() %}
	{{ common.insert_lines('SUPPORT CODE', support_code_lines) }}
{% endmacro %}
