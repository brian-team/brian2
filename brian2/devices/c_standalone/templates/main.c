{# IS_OPENMP_COMPATIBLE #}
#include <stdlib.h>
#include "objects.h"
#include "brianlib/common_math.h"
#include <time.h>
{{ openmp_pragma('include') }}
#include "run.h"

{% for codeobj in code_objects %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

{% for name in additional_headers %}
#include "{{name}}"
{% endfor %}

#include <stdio.h>

{{report_func|autoindent}}

int main(int argc, char **argv)
{
    int i;
	brian_start();

	{
		{{ openmp_pragma('set_num_threads') }}
        {{main_lines|autoindent}}
	}

	brian_end();

	return 0;
}
