flow: main.o flow.o glib.o
	cc  -o flow  main.o flow.o glib.o -lm

matflow: matflow.o matmain.o glib.o
	cc -o matflow matmain.o matflow.o glib.o -lm

matflow.o: matfl.o flow.o
	ld -r matfl.o flow.o -o matflow.o


