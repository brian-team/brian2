{% macro cpp_file() %}
#include "brianlib/SimpleOpt.h"
#include "objects.h"
#include<iostream>
#include <sstream>

enum {
{% for name, param in parameters | dictsort(by='key') %}
    BRIANCMDLINEOPT_{{name}}{% if not loop.last %},{% endif %}
{% endfor %}
    };

void {{simname}}::_set_default_parameters()
{
    // Set parameter values
    {% for name, param in parameters | dictsort(by='key') %}
    _parameter_{{name}} = {{cpp_number_representation(param)}};
    {% endfor %}
    // Initialise maps
    {% for name, param in parameters | dictsort(by='key') %}
    {% set dtype = c_data_type(param.dtype) %}
    _parametermap_{{dtype}}["{{name}}"] = &_parameter_{{name}};
    {% endfor %}
}

{% for dtype in parameter_c_data_types %}
void {{simname}}::_set_parameter_{{dtype}}(std::string name, {{dtype}} value)
{
    *_parametermap_{{dtype}}[name] = value;
}
{% endfor %}

int {{simname}}::_read_command_line_parameters(int argc, char *argv[])
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

{% endmacro %}

{% macro h_file() %}
{% endmacro %}
