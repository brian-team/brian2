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
{% endfor %}

{% for name in user_headers | sort %}
#include {{name}}
{% endfor %}

#include <iostream>
#include <fstream>
#include <string>

{{report_func|autoindent}}

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
