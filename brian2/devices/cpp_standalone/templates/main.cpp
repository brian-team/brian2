#include<time.h>
#include<stdlib.h>
#include "objects.h"
#include<ctime>

{% for codeobj in code_objects %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

#include<iostream>
using namespace std;

int main(void)
{
	clock_t start = clock();
	_init_arrays();
	_load_arrays();
	srand((unsigned int)time(NULL));
	const double dt = {{dt}};
	double t = 0.0;
	{% for main_line in main_lines %}
	{{ main_line }}
	{% endfor %}
	double duration = (clock()-start)/(double)CLOCKS_PER_SEC;
	cout << "Simulation time: " << duration << endl;
	_write_arrays();
	_dealloc_arrays();
	return 0;
}
