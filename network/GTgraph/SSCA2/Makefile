include ../Makefile.var
TARGET = GTgraph-ssca2
TOCOMPILE = graph.o utils.o init.o globals.o drive.o

all: ${TOCOMPILE}
	${CC} $(LDFLAGS) -o $(TARGET) ${TOCOMPILE} ${LIBS}

.c.o:
	$(CC) ${FLAGS} -g ${INCS} -c $*.c

clean:
	rm -f *.o core* *~ sample.gr log $(TARGET)
