{% extends 'common_group.pyx' %}

{% block before_code %}
    _owner.initialise_queue()
{% endblock %}

{% block maincode %}
    _owner.push_spikes()
{% endblock %}
