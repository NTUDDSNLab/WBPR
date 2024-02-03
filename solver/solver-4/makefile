
CFILES1= permute.c maxflow_init.c main.c big_relabel.c maxflow.c queue.c dyn_tree_maxflow.c dyn_splay_maxflow.c fib_heap_maxflow.c netio.c

CFLAGS=-O -D"DOUBLE int"  
DEBUGFILES=

#For compiling with DEBUG:
#CFLAGS=-g -D"DOUBLE int" -DDEBUG
#DEBUGFILES= debug.c

OBJS= $(CFILES1:%.c=%.o) $(DEBUGFILES:%.c=%.o)

CC= gcc

OUTFILE = PLED



LIBS = -lm 

main: ${OBJS} 
	${CC} ${OBJS} ${CFLAGS} ${LDFLAGS} -o ${OUTFILE} ${LIBS}


maxflow_init.o: _maxflow.h maxflow_typedef.h
maxflow.o: _maxflow.h maxflow_typedef.h 
permute.o: _maxflow.h maxflow_typedef.h
main.o: _maxflow.h maxflow_typedef.h
queue.o: queue.h
dyn_tree_maxflow.o: dyn_tree_maxflow.h _dyn_tree_maxflow.h
dyn_splay_maxflow.o: dyn_tree_maxflow.h _dyn_tree_maxflow.h
fib_heap_maxflow.o: maxflow_typedef.h _fib_heap_maxflow.h
netio.o: netio.h _maxflow.h
debug.o: debug.h netio.h

