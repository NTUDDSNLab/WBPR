
CFILES1= genio.c genrmf.c genmain.c

OBJS= $(CFILES1:%.c=%.o)

CC= gcc

OUTFILE = genrmf

CFLAGS=-O 

LIBS = -lm 

main: ${OBJS} 
	${CC} ${OBJS} ${CFLAGS} ${LDFLAGS} -o ${OUTFILE} ${LIBS}
