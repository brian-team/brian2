PROGRAM = main

SRCS = {{source_files}}
{% if not auto_dependencies %}
H_SRCS = {{header_files}}
{% endif %}
OBJS = ${SRCS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
OPTIMISATIONS = {{ compiler_flags }}
CXXFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I. {{ openmp_pragma('compilation') }} {{ compiler_debug_flags }}
LFLAGS = {{ openmp_pragma('compilation') }} {{ linker_flags }} {{ linker_debug_flags }}
{% if auto_dependencies %}
DEPS = ${OBJS:.o=.d}
{% else %}
DEPS = make.deps
{% endif %}

all: $(PROGRAM)

.PHONY: all clean{% if auto_dependencies %} missing-deps{% endif %}

{% if auto_dependencies %}
$(PROGRAM): $(OBJS) makefile
{% else %}
$(PROGRAM): $(OBJS) $(DEPS) makefile
{% endif %}
	$(CXX) $(OBJS) -o $(PROGRAM) $(LFLAGS)

clean:
	{{ rm_cmd }}

{% if auto_dependencies %}
# Recover dependency files deleted independently of their object files.
$(patsubst %.d,%.o,$(filter-out $(wildcard $(DEPS)),$(DEPS))): missing-deps

-include $(DEPS)

%.o : %.cpp makefile
	$(CXX) $(CXXFLAGS) -MMD -MP -MF $(@:.o=.d) -MT $@ $< -o $@

%.o : %.c makefile
	$(COMPILE.c) -MMD -MP -MF $(@:.o=.d) -MT $@ $< -o $@
{% else %}
make.deps: $(SRCS) $(H_SRCS)
	$(CXX) $(CXXFLAGS) -MM $(SRCS) > make.deps

ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.cpp makefile
	$(CXX) $(CXXFLAGS) $< -o $@
{% endif %}
