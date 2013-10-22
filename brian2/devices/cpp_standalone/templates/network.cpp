defaultclock.set_interval(0.0, {{duration}});
while(defaultclock.running())
{
	t = defaultclock.t();
	{% for run_line in run_lines %}
	{{run_line}}
	{% endfor %}
	defaultclock.tick();
}
