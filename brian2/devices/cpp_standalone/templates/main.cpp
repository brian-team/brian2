#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include "run.h"

{% for codeobj in code_objects %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

{% for name in additional_headers %}
#include "{{name}}"
{% endfor %}

#include<iostream>

int main(void)
{
	std::clock_t start = std::clock();

	brian_start();

	{% for main_line in main_lines %}
	{{ main_line }}
	{% endfor %}

	double duration = (std::clock()-start)/(double)CLOCKS_PER_SEC;
	std::cout << "Simulation time: " << duration << endl;

	brian_end();

	return 0;
}
