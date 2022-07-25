#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>
{{ openmp_pragma('include') }}
#include "run.h"
#include "brianlib/common_math.h"
#include "randomkit.h"

{% for codeobj in code_objects | sort(attribute='name') %}
#include "code_objects/{{codeobj.name}}.h"
{% for block in codeobj.before_after_blocks %}
#include "code_objects/{{block}}_{{codeobj.name}}.h"
{% endfor %}
{% endfor %}

{% for name in user_headers | sort %}
#include {{name}}
{% endfor %}

#include <iostream>
#include <fstream>
#include <string>

{{report_func|autoindent}}

void set_from_command_line(int argc, char **argv)
{
    const std::vector<std::string> args(argv + 1, argv + argc);
    for (const auto& arg : args) {
		// Split into two parts
		size_t equal_sign = arg.find("=");
		auto name = arg.substr(0, equal_sign);
		auto value = arg.substr(equal_sign + 1, arg.length());
		brian::set_variable_by_name(name, value);
	}
}
int main(int argc, char **argv)
{
    {{'\n'.join(code_lines['before_start'])|autoindent}}
	brian_start();
    {{'\n'.join(code_lines['after_start'])|autoindent}}
	{
		using namespace brian;

		{{ openmp_pragma('set_num_threads') }}
        {{main_lines|autoindent}}
	}
    {{'\n'.join(code_lines['before_end'])|autoindent}}
	brian_end();
    {{'\n'.join(code_lines['after_end'])|autoindent}}
	return 0;
}
