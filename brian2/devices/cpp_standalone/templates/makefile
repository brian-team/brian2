PROGRAM = main
SRCS = {{source_files}}
H_SRCS = {{header_files}}
OBJS = ${SRCS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
CC = @g++
DEBUG = -g
OPTIMISATIONS = {{ compiler_flags }}
CFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I. {{ openmp_pragma('compilation') }}
LFLAGS = {{ openmp_pragma('compilation') }} {{ linker_flags }}
DEPS = make.deps

all: $(PROGRAM)

# Adds debug flags
debug: CFLAGS += $(DEBUG)
debug: LFLAGS += $(DEBUG)
debug: $(PROGRAM)

.PHONY: all debug clean

$(PROGRAM): $(OBJS) $(DEPS) makefile
	$(CC) $(OBJS) -o $(PROGRAM) $(LFLAGS)

clean:
	{{ rm_cmd }}

make.deps: $(SRCS) $(H_SRCS)
	$(CC) $(CFLAGS) -MM $(SRCS) > make.deps
	
ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.cpp makefile
	$(CC) $(CFLAGS) $< -o $@
