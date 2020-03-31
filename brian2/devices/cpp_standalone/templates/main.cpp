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
{% for block in codeobj.before_after_blocks %}
#include "code_objects/{{block}}_{{codeobj.name}}.h"
{% endfor %}
{% endfor %}

{% for name in user_headers | sort %}
#include {{name}}
{% endfor %}

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


std::string format_time(int time_in_s)
{
	int hours = time_in_s / 3600;
	time_in_s = time_in_s % 3600;
	int minutes = time_in_s / 60;
	time_in_s = time_in_s % 60;

	std::stringstream ss;
	ss << hours;
	std::string hrs = ss.str();

	ss << minutes;
	std::string mins = ss.str();
	
	ss << time_in_s;
	std::string secs = ss.str();

	if(hours < 1){
		if(minutes < 1){
			return secs + " s"
		}else{
			return mins + "m " + secs + "s"; 
		}
	}else{
		return hrs + "h " + mins + "m " + secs + "s";
	}
}

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
