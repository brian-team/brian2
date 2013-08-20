#include "arrays.h"

{% for codeobj in code_objects %}
#include "{{codeobj.name}}.h"
{% endfor %}

int main(void)
{
	const double dt = {{dt}};
	for(int i=0; i<{{num_steps}}; i++)
	{
		double t = i*dt;
		{% for codeobj in code_objects %}
		_run_{{codeobj.name}}(t);
		{% endfor %}
	}
	return 0;
}
