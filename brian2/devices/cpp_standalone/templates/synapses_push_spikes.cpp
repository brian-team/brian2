////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cpp_file() %}
#include "code_objects/{{codeobj_name}}.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>

void _run_{{codeobj_name}}()
{
    using namespace brian;

    {% if openmp_pragma('with_openmp') %}
    const double _start_time = omp_get_wtime();
    {% else %}
    const std::clock_t _start_time = std::clock();
    {% endif %}

    ///// CONSTANTS ///////////
    %CONSTANTS%
    ///// POINTERS ////////////
    {{pointers_lines|autoindent}}

    //// MAIN CODE ////////////
    // we do advance at the beginning rather than at the end because it saves us making
    // a copy of the current spiking synapses
    {#  Get the name of the array that stores these events (e.g. the spikespace array) #}
    {% set _eventspace = get_array_name(eventspace_variable) %}
    {{openmp_pragma('parallel')}}
    {
        {{owner.name}}.advance();
        {{owner.name}}.push({{_eventspace}}, {{_eventspace}}[_num{{eventspace_variable.name}}-1]);
    }

    // Profiling
    {% if openmp_pragma('with_openmp') %}
    const double _run_time = omp_get_wtime() -_start_time;
    {% else %}
    const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    {% endif %}
    {{codeobj_name}}_profiling_info += _run_time;
}
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// HEADER FILE ///////////////////////////////////////////////////////////

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

#include "objects.h"

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
