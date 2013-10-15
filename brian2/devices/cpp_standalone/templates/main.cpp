#include<time.h>
#include<stdlib.h>
#include "objects.h"
#include<ctime>

{% for codeobj in code_objects %}
#include "{{codeobj.name}}.h"
{% endfor %}

#include<iostream>
using namespace std;

int main(void)
{
	clock_t start = clock();
	_init_arrays();
	srand((unsigned int)time(NULL));
	const double dt = {{dt}};
	double t = 0.0;
	{% for main_line in main_lines %}
	{{ main_line }}
	{% endfor %}
	cout << "Num synapses: " << _dynamic_array_synapses__synaptic_pre.size() << endl;
	cout << "Num spikes: " << _dynamic_array_spikemonitor__i.size() << endl;
	double duration = (clock()-start)/(double)CLOCKS_PER_SEC;
	cout << "Time: " << duration << endl;
	_dealloc_arrays();
	return 0;
}
