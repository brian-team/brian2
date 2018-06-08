{% macro cpp_file() %}
#include "parameters.h"

namespace brian {

{% for name, param in parameters | dictsort(by='key') %}
{{c_data_type(param.dtype)}} _parameter_{{name}} = {{cpp_number_representation(param)}};
{% endfor %}

}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_PARAMETERS_H
#define _BRIAN_PARAMETERS_H

namespace brian {

{% for name, param in parameters | dictsort(by='key') %}
extern {{c_data_type(param.dtype)}} _parameter_{{name}};
{% endfor %}

}

#endif

{% endmacro %}
