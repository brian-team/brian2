#include "arrays.h"

{% for codeobj in code_objects %}
#include "{{codeobj.name}}.h"
{% endfor %}

#include<iostream>
using namespace std;

int main(void)
{
	_init_arrays();
	const double dt = {{dt}};
	for(int i=0; i<{{num_steps}}; i++)
	{
		double t = i*dt;
		{% for run_line in run_lines %}
		{{run_line}}
		{% endfor %}
	}
	cout << "Num spikes: " << _dynamic_array_spikemonitor__i.size() << endl;
	_dealloc_arrays();
	return 0;
}
