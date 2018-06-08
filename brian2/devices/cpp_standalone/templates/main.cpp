#include <stdlib.h>
#include "objects.h"
#include "parameters.h"
#include <ctime>
#include <time.h>
{{ openmp_pragma('include') }}
#include "run.h"
#include "brianlib/common_math.h"
#include "randomkit.h"

{% for codeobj in code_objects | sort(attribute='name') %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

{% for name in user_headers | sort %}
#include {{name}}
{% endfor %}

#include <iostream>
#include <fstream>

{{report_func|autoindent}}

void brian::_run_main_lines()
{
    {{ openmp_pragma('set_num_threads') }}
    {{main_lines|autoindent}}
}

int main(int argc, char **argv)
{
    if(read_command_line_parameters(argc, argv)) {
        return 1;
    }
    brian sim;
	sim._start();
	sim._run_main_lines();
	sim._end();

	return 0;
}
