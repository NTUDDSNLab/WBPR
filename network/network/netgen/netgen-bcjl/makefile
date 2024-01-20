CFLAGS = -O -DDIMACS

OBJS = netgen.o index.o random.o

netgen: $(OBJS)
	$(CC) -o $@ $(CFLAGS) $(OBJS)

netgen.o: netgen.c netgen.h

index.o: index.c netgen.h

random.o: random.c
