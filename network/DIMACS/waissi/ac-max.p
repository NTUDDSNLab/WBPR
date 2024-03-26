{
 ACYCLIC NETWORK GENERATOR FOR MAX-FLOW 
                               (revised 11/25/90)
                               (revised 01/04/91)

 Copyright (C) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128

 Tel: (313) 593-5012
 E-mail: gary_waissi@um.cc.mich.edu

 Revisions and corrections: INTEGER changed to LONGINT
 to allow the range [-2^31 ... 2^31-1]. A bug was 
 corrected in the calculation of num_arcs for special
 SPARSE networks, (line 312 shall read:

   for i:=2 to num_nodes do num_arcs:=num_arcs+2;

 This source program has been given to public domain
 through DIMACS. 

 This program generates ACYCLIC MAX-FLOW NETWORKS to 
 user specified files. Two types of acyclic networks
 can be generated: FULLY DENSE and special SPARSE 
 acyclic networks. The two key procedures are:

 PROCEDURE AcyclicNet1;   (fully dense networks)
 PROCEDURE AcyclicNet2;   (special sparse networks)

 Data file format corresponds to DIMACS data file
 specifications described in "The First DIMACS 
 International Algorithm Implementation Challenge:
 Problem Definitions and Specifications". Please 
 contact via e-mail: netflow@dimacs.rutgers.edu

 This program is written in standard Pascal for 
 transportability.

 Another version of this program, that uses windowing
 and has a user-friendly menu interface is available.
 The version uses Turbo Pascal 5.5 features (including
 CTR, DOS), and is compiled in Turbo Pascal Units TPU's.
 Runs on DOS 3.xx and latter versions. Five network
 generators and a new max-flow algorithm are included
 in the package, and can be invoked from the menu.

 This program is robust. A generated arc is written 
 to the disk file, and not stored in the memory. Disk
 full causes a run-time error.
}

Program AcyclicNetwork(input,netres,output);

  type string = packed array [1..20] of char;
  var
    cap:real;
    y,p,q,i,k,num,try,arcnumber:integer;
    node_i,node_j,num_arcs,num_nodes,capacity:integer;
    tail,head:integer;
    ch:char;
    done,special,sparse:boolean;
    OutFileName:string;
    netres:text;
    source:integer;
    sink:integer;

  {Two new Sun-compatible RNG procedures added by McGeoch 11/90}
  Procedure Randomize;
  var i: integer;
    Begin
    	{Pascal function seed(n) sets therng seed to n and returns the
	 previous value.  Integer function wallclock returns the number
	 of seconds elapsed since sometime in 1970. 
	 }
	 i:= seed(wallclock);
     End;
     
  Function irandom(top:integer): integer;
    {Returns an integer from 0..(top-1) }
    Begin
    	{My local Pascal function random(x) ignores its real argument
	 x.  It is not the seed.}
	 irandom := trunc(random(0.0)*top);
    End;

    Procedure WriteArc;
  Begin
    write(netres,'a');
    writeln(netres,tail:10,head:10,cap:10:0);
  End;

  Procedure AcyclicNet1;
    var ss1:real;
    Procedure NewArc1;
    Begin
      if special then
        Begin
          if (head=tail+1) then
            Begin
              ss1:=tail-num_nodes/2;
              cap:=1+sqr(ss1);
            End
          else cap:=1;
        End
      else cap:=random(capacity)+1;
      num:=try;
      try:=try+1;
      WriteArc;
    End;

  Begin                  { Procedure AcyclicNet1 }
    k:=1;   
    for p:=1 to (num_nodes-1) do
    Begin
      tail:=p;
      for q:=p+1 to num_nodes do
      Begin
        head:=q;
        NewArc1;
      End;
    End;
  End;

  Procedure AcyclicNet2;
    Procedure NewArc2;
    Begin
      if (head=tail+1) and (head<>num_nodes) then
        cap:=num_nodes
      else cap:=1;
      num:=try;
      try:=try+1;
      WriteArc;
    End;

  Begin                  { Procedure AcyclicNet2 }
    k:=1;   
    for p:=1 to (num_nodes-1) do
    Begin
      tail:=p;
      head:=p+1;
      NewArc2;
      if (head<>num_nodes) then
      Begin
        tail:=p;
        head:=num_nodes;
        NewArc2;
      End;
    End;
  End;

  Procedure Banner1;
  Begin
    writeln;
    writeln('     ACYCLIC MAX-FLOW NETWORK');
    writeln('           GENERATOR');
    writeln('    Copyright (C) Gary R. Waissi');
    writeln('              (1990)');
    writeln('   University of Michigan-Dearborn');
    writeln('     School of Management, 113FOB');
    writeln('        Dearborn, MI 48128');
    writeln;
    writeln('  The program generates an ACYCLIC');
    writeln('  MAX-FLOW network into a user file.');
    writeln('  An acyclic network is such that');
    writeln('  for each arc (i,j), where i is the');
    writeln('  arc tail node number, and j is the');
    writeln('  arc head node number i<j.');
    writeln;
    writeln('  The generated acyclic network will');
    writeln('  be either fully dense or sparse.');
    writeln;
  End;
  
  Procedure Banner2;
    Begin
      writeln;
      writeln('  This program can generate two');
      writeln('  types of acyclic networks: FULLY');
      writeln('  DENSE and special SPARSE ACYCLIC');
      writeln('  networks.');
      writeln; 
      writeln('  For the FULLY DENSE acyclic ');
      writeln('  networks the user has two options');
      writeln('  for arc flow capacities: random');
      writeln('  or special capacities. Special ');
      writeln('  arcs capacities are calculated ');
      writeln('  using the concept of Glover et al.');
      writeln('  published in "A Comprehensive  ');
      writeln('  Computer Evaluation and Enhancement');
      writeln('  of Maximum Flow Algorithms", Appl.');
      writeln('  of MS, Vol 3, 1983. This approach');
      writeln('  forces flow on each to capacity at');
      write('  Press ENTER <ret> to continue');
      readln;
      writeln;
      writeln('  optimum. The networks were called');
      writeln('  in the study "hard networks", ');
      writeln('  because such networks were found');
      writeln('  to cause algorithms to their worst');
      writeln('  case performance.');
      writeln('  The special SPARSE acyclic networks');
      writeln('  are such that each node is connected');
      writeln('  by an arc to the next node and to');
      writeln('  the sink node, i.e. there exist two');
      writeln('  types of arcs: 1. (i,j) where j=i+1');
      writeln('  for all i and j\t with capacity n;');
      writeln('  and 2. (i,t) for all i\t with ');
      writeln('  capacity 1. These simple networks');
      writeln('  cause the Dinic Algorithm (Dinic,');
      writeln('  E.A., Algorithm for Solution of ');
      writeln('  a Problem of Maximum Flow in ');
      write('  Press ENTER <ret> to continue');
      readln;
      writeln;
      writeln('  a Network with Power Estimation,');
      writeln('  Soviet Math. Dokl. Vol. 11, No. 5,');
      writeln('  pp. 1277-1280, 1970), to it''s worst');
      writeln('  case performance. These sparse ');
      writeln('  acyclic networks were presented in');
      writeln('  the dissertation of Gary R. Waissi,');
      writeln('  "Acyclic Network Generation and');
      writeln('  Maximal Flow Algorithms for Single');
      writeln('  Commodity Flow", University of');
      writeln('  Michigan, 1985.');
      write('  Press ENTER <ret> to continue');
      readln;
    End;
  
  Procedure Banner3;
    Begin
      writeln;
      writeln('  You can choose either random');
      writeln('  capacities for arcs, or special');
      writeln('  arc capacities. Special arc ');
      writeln('  capacities are calculated using');
      writeln('  the following (Glover et al):');
      writeln('  1. cap(i,j)=1 for each (i,j) with');
      writeln('     j>i+1');
      writeln('  2. cap(i,j)=1+(i-n/2)^2 for each');
      writeln('     (i,j) with j=i+1');
      writeln;
      writeln('  The program, however, will round');
      writeln('  all non-integer capacities to');
      writeln('  integer values to conform with');
      writeln('  DIMACS specifications.');
      writeln;
    End;

  Procedure Banner4;
    Begin
      if not sparse then
        writeln(netres,'c Fully Dense Acyclic Network')
      else writeln(netres,'c Sparse Acyclic Network');
      writeln(netres,'c for Max-Flow');
      if not special then
        writeln(netres,'c Arcs with random capacities')
      else writeln(netres,'c Arcs with special capacities');
      writeln(netres,'p max',num_nodes:10,num_arcs:10);
      writeln(netres,'n',source:14,'  s');
      writeln(netres,'n',sink:14,'  t');
      writeln;
      writeln('  Please notice, that for simplicity');
      writeln('  the source node s is always numbered');
      writeln('  with 1, and the sink node t is');
      writeln('  assigned the largest node number,');
      writeln('  here t = ',sink,'.');
      write('  Press ENTER <ret> to continue');
      readln;
      writeln;
      writeln('  AN ACYCLIC MAX-FLOW NETWORK will');
      writeln('  be generated to file ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The sink node is node   ',sink:8);
      write('  Press ENTER <ret> to continue');
      readln;
      writeln;
      writeln('  The network is given as a list of');
      writeln('  lines: comment lines, problem line,');
      writeln('  node lines, arc lines as follows:');
      writeln('  c this is comment line');
      writeln('  c The "p" line gives the problem');
      writeln('  c TYPE, number of nodes, and ');
      writeln('  c number of arcs. Two "n" lines');
      writeln('  c list the source node, and the');
      writeln('  c sink node, one line per node,');
      writeln('  c including a node designator (s,t).');
      writeln('  c The "a" lines list the arcs, one');
      writeln('  c line per arc with three numbers:');
      writeln('  c   Source Destination Capacity');
      writeln('  c For example like this:');
      writeln('  p MAX NODES ARCS');
      writeln('  n NODE WHICH');
      writeln('  a FROM TO CAPACITY');
      write('  Press ENTER <ret> to continue');
      readln;
    End;

  Procedure Banner5;
    Begin
      writeln('  The network is completed. The data');
      writeln('  file is an ASCII file, with name:');
	  writeln('  ',OutFileName,',and can be');
      writeln('  accessed with your editor.');
    End;

  Procedure UserValues;
    Begin
      sparse:=true;
      special:=false;
      done:=false;
      try:=1;
      writeln;
      repeat
        write('  How many nodes are in the network: ');
        readln(num_nodes);
        if (num_nodes<=1) then
        writeln('  Try again. Need at least two nodes.');
      until (num_nodes>1);
      source:=1;
      sink:=num_nodes;
      Banner2;
      writeln('  Do you want a FULLY DENSE or');
      writeln('  a SPARSE acyclic network.');
      writeln('  Type F for FULLY DENSE and S for');
      write('  SPARSE (F/S): ');
      readln(ch);
      if (ch='f') or (ch='F') then
      Begin
        sparse:=false;
        num_arcs:=0;
        for i:=1 to (num_nodes-1) do num_arcs:=num_arcs+i;
        Banner3;
        writeln('  Do you want special arc flow');
        write('  capacities (Y/N): ');
        readln(ch);
        if (ch='y') or (ch='Y') then special:=true
        else
          begin
            writeln('  What is the upper bound for arc');
            write('  FLOW CAPACITY: ');
            readln(capacity);
          end;
        Banner4;
        AcyclicNet1;
      End
      else if (ch='s') or (ch='S') then
      Begin
        num_arcs:=-1;
        for i:=2 to num_nodes do num_arcs:=num_arcs+2;
        special:=true;
        Banner4;
        AcyclicNet2;
      End;
    End;

Begin
  randomize;  {Turbo Pascal random number generator initiator.  }
              {You may have to replace this by compiler         }
              {specific randomizer, or write a short procedure. }
  Banner1;
  writeln('  Try the network generation out by');
  writeln('  following the instructions.');
  write('  Do you want to continue (Y/N) ');
  readln(ch);
  if (ch='y') or (ch='Y') then
  Begin
    write('  Enter name of the output file: ');
    readln(OutFileName);
{ SUN     assign(netres,OutFileName+'.max'); }
    rewrite(netres,OutFileName);
    UserValues;
{ SUN     close(netres); }
    Banner5;
  End;
End.
