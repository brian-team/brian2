{% macro cpp_file() %}
#include "parameters.h"
#include "brianlib/SimpleOpt.h"
#include<iostream>
#include <sstream>

namespace brian {

{% for name, param in parameters | dictsort(by='key') %}
{{c_data_type(param.dtype)}} _parameter_{{name}} = {{cpp_number_representation(param)}};
{% endfor %}

enum {
{% for name, param in parameters | dictsort(by='key') %}
    BRIANCMDLINEOPT_{{name}}{% if not loop.last %},{% endif %}
{% endfor %}
    };

int read_command_line_parameters(int argc, char *argv[])
{
    CSimpleOpt::SOption options[] = {
{% for name, param in parameters | dictsort(by='key') %}
        { BRIANCMDLINEOPT_{{name}}, "-{{name}}", SO_REQ_CMB },
{% endfor %}
        SO_END_OF_OPTIONS
        };
    CSimpleOpt args(argc, argv, options);
    while (args.Next())
    {
        if (args.LastError() == SO_SUCCESS) {
            switch(args.OptionId()) {
                {% for name, param in parameters | dictsort(by='key') %}
                case BRIANCMDLINEOPT_{{name}}:
                    std::stringstream string_stream(args.OptionArg());
                    string_stream >> _parameter_{{name}};
                    break;
                {% endfor %}
            }
        }
        else {
            std::cout << "Invalid argument: " << args.OptionText() << std::endl;
            return 1;
        }
    }
    return 0;
};

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

int read_command_line_parameters(int argc, char *argv[]);

}

#endif

{% endmacro %}
