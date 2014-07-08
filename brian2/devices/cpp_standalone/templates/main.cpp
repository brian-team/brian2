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

int main(int argc, char **argv)
{
	std::clock_t start = std::clock();

	brian_start();

	{
		using namespace brian;
        {{main_lines|autoindent}}
	}

	double _run_duration = (std::clock()-start)/(double)CLOCKS_PER_SEC;
	std::cout << "Simulation time: " << _run_duration << endl;

	brian_end();

	return 0;
}
