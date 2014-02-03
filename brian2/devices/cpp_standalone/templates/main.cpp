#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>
#include <omp.h>
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

	timespec start, stop;
	clock_gettime(CLOCK_REALTIME, &start);
	//std::clock_t start = std::clock();

	brian_start();

	{
		using namespace brian;
        {{main_lines|autoindent}}
	}

	clock_gettime(CLOCK_REALTIME, &stop);
	double _run_duration = ( stop.tv_sec - start.tv_sec ) + (( stop.tv_nsec - start.tv_nsec ) / 1e9);
	//double _run_duration = (std::clock()-start)/(double)CLOCKS_PER_SEC;
	std::cout << "Simulation time: " << _run_duration << endl;

	brian_end();

	return 0;
}
