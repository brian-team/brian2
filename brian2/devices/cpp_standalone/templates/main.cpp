#include<stdlib.h>
#include "objects.h"
#include<ctime>

{% for codeobj in code_objects %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

#include<iostream>

int main(void)
{
	std::clock_t start = std::clock();
	_init_arrays();
	_load_arrays();
	srand((unsigned int)time(NULL));

	{% for main_line in main_lines %}
	{{ main_line }}
	{% endfor %}

	double duration = (std::clock()-start)/(double)CLOCKS_PER_SEC;
	std::cout << "Simulation time: " << duration << endl;

	_write_arrays();
	_dealloc_arrays();

	return 0;
}
