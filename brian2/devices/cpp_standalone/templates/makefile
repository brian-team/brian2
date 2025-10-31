PROGRAM = main

SRCS = {{source_files}}
H_SRCS = {{header_files}}
OBJS = ${SRCS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
OPTIMISATIONS = {{ compiler_flags }}
CXXFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I. {{ openmp_pragma('compilation') }} {{ compiler_debug_flags }}
LFLAGS = {{ openmp_pragma('compilation') }} {{ linker_flags }} {{ linker_debug_flags }}
DEPS = make.deps

all: $(PROGRAM)

.PHONY: all clean

$(PROGRAM): $(OBJS) $(DEPS) makefile
	$(CXX) $(OBJS) -o $(PROGRAM) $(LFLAGS)

clean:
	{{ rm_cmd }}

make.deps: $(SRCS) $(H_SRCS)
	$(CXX) $(CXXFLAGS) -MM $(SRCS) > make.deps

ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.cpp makefile
	$(CXX) $(CXXFLAGS) $< -o $@
