all: dinic gold_q gold_hlf flow
dinic: augment.o main_dinic.o dinic.o goldberg.o netflow.h io.o utility.o manip.o random.o timer.o
	cc -O  -p -o dinic dinic.o augment.o goldberg.o main_dinic.o io.o utility.o manip.o  random.o timer.o
gold_q: augment.o main_gold_q.o dinic.o goldberg.o netflow.h io.o utility.o manip.o random.o timer.o
	cc -O  -p -o gold_q dinic.o augment.o goldberg.o main_gold_q.o io.o utility.o manip.o  random.o timer.o
gold_hlf: augment.o main_gold_hlf.o dinic.o goldberg.o netflow.h io.o utility.o manip.o random.o timer.o
	cc -O  -p -o gold_hlf dinic.o augment.o goldberg.o main_gold_hlf.o io.o utility.o manip.o  random.o timer.o

gold_hlf2: augment.o main_gold_hlf.o dinic.o goldberg.o netflow.h io.o utility.o manip.o random.o timer.o
	cc -g  -o gold_hlf2 dinic.o augment.o goldberg.o main_gold_hlf.o io.o utility.o manip.o  random.o timer.o

flow: augment.o netflow.o dinic.o goldberg.o netflow.h io.o utility.o manip.o random.o timer.o
	cc -O  -p -o flow dinic.o augment.o goldberg.o netflow.o io.o utility.o manip.o  random.o timer.o
io.o : ghead.h io.c
	cc -O -c  io.c
utility.o : ghead.h utility.c
	cc -O -c  utility.c
manip.o : ghead.h manip.c
	cc -O -c  manip.c
random.o : ghead.h random.c
	cc -O -c  random.c
timer.o : ghead.h timer.c
	cc -O -c  timer.c
netflow.o : netflow.h netflow.c
	cc -O -c  netflow.c
dinic.o : netflow.h dinic.c
	cc -O -c  dinic.c
augment.o : netflow.h augment.c
	cc -O -c  augment.c
goldberg.o : netflow.h goldberg.c
	cc -O -c   goldberg.c
main_gold_hlf.o :
	cc -O -c   main_gold_hlf.c

