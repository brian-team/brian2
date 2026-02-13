
{% set _safe_name = codeobj_name | replace(".", "_") | replace("*", "") | replace("-", "_") %}

{# ── Helper: build the parameter list for a C++ function signature ── #}
{% macro param_list() %}
{% for c_type, param_name, ns_key in function_params %}{{ c_type }} {{ param_name }}{% if not loop.last %}, {% endif %}{% endfor %}
{% endmacro %}


{# ══════════════════════════════════════════════════════════════════════ #}
{# BLOCK: before_run — runs once before simulation starts               #}
{# ══════════════════════════════════════════════════════════════════════ #}
{% macro before_run() %}
{% set _func_name = "_brian_cppyy_before_run_" + _safe_name %}

// Per-codeobject support code (user functions, hashdefines)
{{ hashdefine_lines }}
{{ support_code_lines }}

extern "C" void {{ _func_name }}({{ param_list() }}) {
    {{ denormals_code_lines }}
    {% block before_code %}
    // EMPTY_CODE_BLOCK
    {% endblock %}
}
{% endmacro %}


{# ══════════════════════════════════════════════════════════════════════ #}
{# BLOCK: run — the main simulation step, runs every timestep           #}
{# ══════════════════════════════════════════════════════════════════════ #}
{% macro run() %}
{% set _func_name = "_brian_cppyy_run_" + _safe_name %}

// Per-codeobject support code
{{ hashdefine_lines }}
{{ support_code_lines }}

// Template-specific support code (e.g. synaptic queue access)
{% block template_support_code %}
{% endblock %}

extern "C" void {{ _func_name }}({{ param_list() }}) {
    {{ denormals_code_lines }}
    {% block maincode %}
    {% endblock %}
}
{% endmacro %}


{# ══════════════════════════════════════════════════════════════════════ #}
{# BLOCK: after_run — runs once after simulation completes               #}
{# ══════════════════════════════════════════════════════════════════════ #}
{% macro after_run() %}
{% set _func_name = "_brian_cppyy_after_run_" + _safe_name %}

// Per-codeobject support code
{{ hashdefine_lines }}
{{ support_code_lines }}

extern "C" void {{ _func_name }}({{ param_list() }}) {
    {{ denormals_code_lines }}
    {% block after_code %}
    // EMPTY_CODE_BLOCK
    {% endblock %}
}
{% endmacro %}
