PROGRAM = main
SRCS = {{source_files}}
H_SRCS = {{header_files}}
OBJS = ${SRCS:.cpp=.o}
CC = @g++
DEBUG = -g
OPTIMISATIONS = {{ compiler_flags }}
CFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I. {{ openmp_pragma('compilation') }}
LFLAGS = {{ openmp_pragma('compilation') }}
DEPS = make.deps

all: executable

# Adds debug flags
debug: CFLAGS += $(DEBUG)
debug: LFLAGS += $(DEBUG)
debug: executable

# Adds -march=native to optimisations, which only works for recent gcc versions
native: OPTIMISATIONS += -march=native
native: executable

.PHONY: all debug native executable clean

executable: $(OBJS) $(DEPS)
	$(CC) $(LFLAGS) $(OBJS) -o $(PROGRAM)

clean:
	{{ rm_cmd }}

make.deps: $(SRCS) $(H_SRCS)
	$(CC) $(CFLAGS) -MM $(SRCS) > make.deps
	
ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@
