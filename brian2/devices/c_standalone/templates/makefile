PROGRAM = main
SRCS = {{source_files}}
H_SRCS = {{header_files}}
OBJS = ${SRCS:.c=.o}
CC = @gcc
DEBUG = -g
OPTIMISATIONS = {{ compiler_flags }}
CFLAGS = -std=c99 -c -Wno-write-strings $(OPTIMISATIONS) -I. {{ openmp_pragma('compilation') }}
LFLAGS = -lm {{ openmp_pragma('compilation') }}
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
	$(CC) $(OBJS) -o $(PROGRAM) $(LFLAGS)

clean:
	{{ rm_cmd }}

make.deps: $(SRCS) $(H_SRCS)
	$(CC) $(CFLAGS) -MM $(SRCS) > make.deps
	
ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.c
	$(CC) $(CFLAGS) $< -o $@
