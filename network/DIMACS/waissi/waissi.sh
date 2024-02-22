:	This is a shell archive.
:	Remove everything above this line and
:	run the following text with /bin/sh to create:
:	waissi/ac-max.p
:	waissi/ac-max.pas
:	waissi/bi-max.p
:	waissi/bi-max.pas
:	waissi/index
:	waissi/ra-max.p
:	waissi/ra-max.pas
:	waissi/random.p
:	waissi/random.pas
:	waissi/readme
:	waissi/readme_next
:	waissi/tp55
:	waissi/tr1-max.p
:	waissi/tr1-max.pas
:	waissi/tr2-max.p
:	waissi/tr2-max.pas
: This archive created: Fri Sep 13 10:13:51 1991
cat << 'SHAR_EOF' > waissi/ac-max.p
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
SHAR_EOF
cat << 'SHAR_EOF' > waissi/ac-max.pas
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

Program AcyclicNetwork(input,netres);

  var
    cap:real;
    y,p,q,i,k,num,try,arcnumber:longint;
    node_i,node_j,num_arcs,num_nodes,capacity:longint;
    tail,head:longint;
    ch:char;
    done,special,sparse:boolean;
    OutFileName:string[20];
    netres:text;
    source:longint;
    sink:longint;

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
      writeln('  file is an ASCII file, and can be');
      writeln('  accessed with your editor. You may');
      writeln('  print the file using the DOS');
      writeln('  command PRINT ',OutFileName,'.max.');
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
    assign(netres,OutFileName+'.max');
    rewrite(netres);
    UserValues;
    close(netres);
    Banner5;
  End;
End.
SHAR_EOF
cat << 'SHAR_EOF' > waissi/bi-max.p
{
 BIPARTITE MAX-FLOW NETWORK GENERATOR FOR MAX-FLOW
                                (revised 11/25/90)
                                (revised 01/04/91)

 Copyright (C) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128

 Tel: (313) 593-5012
 E-mail: gary_waissi@um.cc.mich.edu

 Revisions: INTEGER is changed to LONGINT to allow
 range [-2^31 ... 2^31-1]. 

 This program generates BIPARTITE MAX-FLOW NETWORKS
 to user specified files. The key procedure is:

 PROCEDURE BipartiteNet;

 Data file format corresponds to DIMACS data file
 specifications described in "The First DIMACS 
 International Algorithm Implementation Challenge: 
 Problem Definitions and Specifications". Please 
 contact via e-mail: netflow@dimacs.rutgers.edu

 This program is written in standard Pascal for
 transportability.

 Another version, that uses windowing and a user
 friendly menu interface is available. The version
 uses Turbo Pascal 5.5 features (including CTR, DOS),
 and is compiled in Turbo Pascal Units TPU's. Runs on
 DOS 3.xx and latter versions. Five network generators
 and a new max-flow algorithm are included in the 
 package, and can be invoked from the menu.

 This program is robust. A generated arc is written 
 to the disk file, and not stored in the memory. Disk
 full causes a run-time error.
}

Program BipartiteNetwork(input,output);

  type string = packed array [1..20] of char;
  var 
      ii,jj,cap:integer;
      i,k,kk,z,num,col,imax,jmax,capacity:integer;
      num_nodes,num_arcs:integer;
      source,sink:integer;
      ch:char;
      found:boolean;
      unit_cap:boolean;
      OutFileName:string;
      netres:text;

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
  Procedure WriteArcs;
    Begin
      write(netres,'a');
      writeln(netres,ii:10,jj:10,cap:10);
    End;

  Procedure BipartiteNet;
    Procedure SourceArcs;
      Begin
        i:=2; z:=1;
        repeat
          ii:=1; jj:=i;
          cap:=irandom(capacity)+1;
          WriteArcs;
          z:=z+1; i:=i+1;
        until i=imax+2;
      End;

    Procedure BipartiteArcs;
      Begin
        i:=2;
        k:=imax+1;
        repeat
          repeat
            ii:=i;
            jj:=k+1;
            if unit_cap then cap:=1
            else cap:=irandom(capacity)+1;
            WriteArcs;
            z:=z+1;
            k:=k+1;
          until k=imax+jmax+1;
          i:=i+1;
          k:=imax+1;
        until i=imax+2;
      End;

    Procedure SinkArcs;
      Begin
        i:=imax+2;
        repeat
          ii:=i;
          jj:=imax+jmax+2;
          cap:=irandom(capacity)+1;
          WriteArcs;
          z:=z+1; i:=i+1;
        until i=imax+jmax+2;
      End;

    Begin                 { Procedure BipartiteNet; }
      kk:=1;
      SourceArcs;
      BipartiteArcs;
      SinkArcs;
    End;
  
  Procedure Banner1;
    Begin
      writeln;
      writeln('    BIPARTITE MAX-FLOW NETWORK');
      writeln('           GENERATOR');
      writeln('    Copyright (C) Gary R. Waissi');
      writeln('              (1990)');
      writeln('   University of Michigan-Dearborn');
      writeln('     School of Management, 113FOB');
      writeln('        Dearborn, MI 48128');
      writeln;
      writeln('  This program generates a BIPARTITE');
      writeln('  MAX-FLOW network into a user file.');
      writeln('  The bipartite network is appended');
      writeln('  with two nodes, a common source ');
      writeln('  "super source" and a common sink');
      writeln('  "super sink". The program is');
      writeln('  designed for max-flow applications.');
      writeln('  The generated data file corresponds');
      writeln('  to DIMACS (1990) specifications.');
      writeln;
    End;

  Procedure Banner2;
    Begin
      writeln;
      writeln('  A BIPARTITE MAX-FLOW NETWORK will');
      writeln('  be generated to file ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The sink node is node   ',source:8);
      writeln('  The sink node is numbered with the ');
      writeln('  largest node number. ');
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
      writeln;
      writeln(netres,'c Bipartite Network for Max-Flow');
      if unit_cap then
        writeln(netres,'c with UNIT capacities on bipartite arcs')
      else
        writeln(netres,'c with RANDOM capacities');
      writeln(netres,'p max',num_nodes:10,num_arcs:10);
      writeln(netres,'n',source:14,'  s');
      writeln(netres,'n',sink:14,'  t');
      writeln;
      writeln('  The network generation is in progress.');
      writeln('  Please wait.');
    End;

  Procedure Banner3;
    Begin
      writeln;
      writeln('  The network is completed. The data');
      writeln('  file is an ASCII file, with name:');
	  writeln('  ',OutFileName,',and can be');
      writeln('  accessed with your editor.');
    End;

  Procedure UserValues;
    Begin
      unit_cap:=false;
      repeat
        writeln('  Type the number of nodes on');
        write('  the source side: ');
        readln(imax);
        if (imax<1) then
        writeln('  Try again. Need at least one node.');
      until (imax>=1);
      repeat
        writeln('  Type the number of nodes on the');
        write('  sink side: ');
        readln(jmax);
        if (jmax<1) then
        writeln('  Try again. Need at least one node.');
      until (jmax>=1);
      writeln;
      writeln('  You have two choices for arc');
      writeln('  capacities: either unit capacities');
      writeln('  or random capacities.');
      writeln('  Do you want unit flow capacities');
      write('  on bipartite arcs (Y/N) ');
      readln(ch);
      if (ch='y') or (ch='Y') then unit_cap:=true;
      if unit_cap then
        Begin
          writeln('  Give an upper bound for the ');
          writeln('  random CAPACITY for arcs out');
          write('  of the source and into the sink: ');
          readln(capacity);
        End
      else
        Begin
          writeln('  Give an upper bound for the');
          write('  random CAPACITY: ');
          readln(capacity);
        End;
      num_nodes:=imax+jmax+2;
      num_arcs:=imax+jmax+imax*jmax;
      source:=1;
      sink:=num_nodes;
    End;

  Begin
    randomize;  {Turbo Pascal random number generator initiator.}
                {You may to replace this by compiler specific   }
                {randomizer, or write a short procedure.        }
    Banner1;
    writeln('  Try the network generation out by');
    writeln('  following the instructions.');
    write('  Do you want to continue (Y/N) ');
    readln(ch);
    if (ch='y') or (ch='Y') then
    Begin
      write('  Enter name of the output file: ');
      readln(OutFileName);
{ SUN       assign(netres,OutFileName+'.max'); }
      rewrite(netres,OutFileName);
      UserValues;
      Banner2;
      BipartiteNet;
{ SUN       close(netres); }
      Banner3;
    End;
  End.
SHAR_EOF
cat << 'SHAR_EOF' > waissi/bi-max.pas
{
 BIPARTITE MAX-FLOW NETWORK GENERATOR FOR MAX-FLOW
                                (revised 11/25/90)
                                (revised 01/04/91)

 Copyright (C) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128

 Tel: (313) 593-5012
 E-mail: gary_waissi@um.cc.mich.edu

 Revisions: INTEGER is changed to LONGINT to allow
 range [-2^31 ... 2^31-1]. 

 This program generates BIPARTITE MAX-FLOW NETWORKS
 to user specified files. The key procedure is:

 PROCEDURE BipartiteNet;

 Data file format corresponds to DIMACS data file
 specifications described in "The First DIMACS 
 International Algorithm Implementation Challenge: 
 Problem Definitions and Specifications". Please 
 contact via e-mail: netflow@dimacs.rutgers.edu

 This program is written in standard Pascal for
 transportability.

 Another version, that uses windowing and a user
 friendly menu interface is available. The version
 uses Turbo Pascal 5.5 features (including CTR, DOS),
 and is compiled in Turbo Pascal Units TPU's. Runs on
 DOS 3.xx and latter versions. Five network generators
 and a new max-flow algorithm are included in the 
 package, and can be invoked from the menu.

 This program is robust. A generated arc is written 
 to the disk file, and not stored in the memory. Disk
 full causes a run-time error.
}

Program BipartiteNetwork(input,output);

  var 
      ii,jj,cap:longint;
      i,k,kk,z,num,col,imax,jmax,capacity:longint;
      num_nodes,num_arcs:longint;
      source,sink:longint;
      ch:char;
      found:boolean;
      unit_cap:boolean;
      OutFileName:string[20];
      netres:text;

  Procedure WriteArcs;
    Begin
      write(netres,'a');
      writeln(netres,ii:10,jj:10,cap:10);
    End;

  Procedure BipartiteNet;
    Procedure SourceArcs;
      Begin
        i:=2; z:=1;
        repeat
          ii:=1; jj:=i;
          cap:=random(capacity)+1;
          WriteArcs;
          z:=z+1; i:=i+1;
        until i=imax+2;
      End;

    Procedure BipartiteArcs;
      Begin
        i:=2;
        k:=imax+1;
        repeat
          repeat
            ii:=i;
            jj:=k+1;
            if unit_cap then cap:=1
            else cap:=random(capacity)+1;
            WriteArcs;
            z:=z+1;
            k:=k+1;
          until k=imax+jmax+1;
          i:=i+1;
          k:=imax+1;
        until i=imax+2;
      End;

    Procedure SinkArcs;
      Begin
        i:=imax+2;
        repeat
          ii:=i;
          jj:=imax+jmax+2;
          cap:=random(capacity)+1;
          WriteArcs;
          z:=z+1; i:=i+1;
        until i=imax+jmax+2;
      End;

    Begin                 { Procedure BipartiteNet; }
      kk:=1;
      SourceArcs;
      BipartiteArcs;
      SinkArcs;
    End;
  
  Procedure Banner1;
    Begin
      writeln;
      writeln('    BIPARTITE MAX-FLOW NETWORK');
      writeln('           GENERATOR');
      writeln('    Copyright (C) Gary R. Waissi');
      writeln('              (1990)');
      writeln('   University of Michigan-Dearborn');
      writeln('     School of Management, 113FOB');
      writeln('        Dearborn, MI 48128');
      writeln;
      writeln('  This program generates a BIPARTITE');
      writeln('  MAX-FLOW network into a user file.');
      writeln('  The bipartite network is appended');
      writeln('  with two nodes, a common source ');
      writeln('  "super source" and a common sink');
      writeln('  "super sink". The program is');
      writeln('  designed for max-flow applications.');
      writeln('  The generated data file corresponds');
      writeln('  to DIMACS (1990) specifications.');
      writeln;
    End;

  Procedure Banner2;
    Begin
      writeln;
      writeln('  A BIPARTITE MAX-FLOW NETWORK will');
      writeln('  be generated to file ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The sink node is node   ',source:8);
      writeln('  The sink node is numbered with the ');
      writeln('  largest node number. ');
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
      writeln;
      writeln(netres,'c Bipartite Network for Max-Flow');
      if unit_cap then
        writeln(netres,'c with UNIT capacities on bipartite arcs')
      else
        writeln(netres,'c with RANDOM capacities');
      writeln(netres,'p max',num_nodes:10,num_arcs:10);
      writeln(netres,'n',source:14,'  s');
      writeln(netres,'n',sink:14,'  t');
      writeln;
      writeln('  The network generation is in progress.');
      writeln('  Please wait.');
    End;

  Procedure Banner3;
    Begin
      writeln;
      writeln('  The network is completed. The data');
      writeln('  file is an ASCII file, and can be');
      writeln('  accessed with your editor. You may');
      writeln('  print the file using the DOS');
      writeln('  command PRINT ',OutFileName,'.max.');
    End;

  Procedure UserValues;
    Begin
      unit_cap:=false;
      repeat
        writeln('  Type the number of nodes on');
        write('  the source side: ');
        readln(imax);
        if (imax<1) then
        writeln('  Try again. Need at least one node.');
      until (imax>=1);
      repeat
        writeln('  Type the number of nodes on the');
        write('  sink side: ');
        readln(jmax);
        if (jmax<1) then
        writeln('  Try again. Need at least one node.');
      until (jmax>=1);
      writeln;
      writeln('  You have two choices for arc');
      writeln('  capacities: either unit capacities');
      writeln('  or random capacities.');
      writeln('  Do you want unit flow capacities');
      write('  on bipartite arcs (Y/N) ');
      readln(ch);
      if (ch='y') or (ch='Y') then unit_cap:=true;
      if unit_cap then
        Begin
          writeln('  Give an upper bound for the ');
          writeln('  random CAPACITY for arcs out');
          write('  of the source and into the sink: ');
          readln(capacity);
        End
      else
        Begin
          writeln('  Give an upper bound for the');
          write('  random CAPACITY: ');
          readln(capacity);
        End;
      num_nodes:=imax+jmax+2;
      num_arcs:=imax+jmax+imax*jmax;
      source:=1;
      sink:=num_nodes;
    End;

  Begin
    randomize;  {Turbo Pascal random number generator initiator.}
                {You may to replace this by compiler specific   }
                {randomizer, or write a short procedure.        }
    Banner1;
    writeln('  Try the network generation out by');
    writeln('  following the instructions.');
    write('  Do you want to continue (Y/N) ');
    readln(ch);
    if (ch='y') or (ch='Y') then
    Begin
      write('  Enter name of the output file: ');
      readln(OutFileName);
      assign(netres,OutFileName+'.max');
      rewrite(netres);
      UserValues;
      Banner2;
      BipartiteNet;
      close(netres);
      Banner3;
    End;
  End.
SHAR_EOF
cat << 'SHAR_EOF' > waissi/index
total 142
-rw-rw-r--  1 badics      12256 Feb 21  1991 ac-max.p
-rw-rw-r--  1 badics      11672 Feb 21  1991 ac-max.pas
-rw-rw-r--  1 badics       8523 Feb 21  1991 bi-max.p
-rw-rw-r--  1 badics       7946 Feb 21  1991 bi-max.pas
-r--rw-r--  1 badics          0 Sep 12 21:01 index
-rw-rw-r--  1 badics      11614 Feb 21  1991 ra-max.p
-rw-rw-r--  1 badics      11027 Feb 21  1991 ra-max.pas
-rw-rw-r--  1 mcgeoch     11098 Nov 30  1990 random.p
-rw-rw-r--  1 mcgeoch     10563 Nov 30  1990 random.pas
-rw-rw-r--  1 mcgeoch      1195 Jul  1 19:28 readme
-rw-rw-r--  1 badics      18202 Feb 21  1991 readme_next
drwxrwsr-x  2 badics        512 May 29 19:54 tp55
-rw-rw-r--  1 badics       8781 Feb 21  1991 tr1-max.p
-rw-rw-r--  1 badics       8204 Feb 21  1991 tr1-max.pas
-rw-rw-r--  1 badics       8568 Feb 21  1991 tr2-max.p
-rw-rw-r--  1 badics       7989 Feb 21  1991 tr2-max.pas
SHAR_EOF
cat << 'SHAR_EOF' > waissi/ra-max.p
{
 RANDOM NETWORK GENERATOR FOR MAX-FLOW  (revised 11/25/90)

 Copyright (C) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128

 Tel: (313) 593-5012
 E-mail: gary_waissi@um.cc.mich.edu

 Revisions: INTEGER changed to LONGINT to allow the
 range of [-2^31 ... 2^31-1].

 This program generates RANDOM NETWORKS for MAX-FLOW
 applications to user specified files. Key procedure is:

 PROCEDURE RandomNet;

 Data file format corresponds to DIMACS data file
 specifications described in "The First DIMACS 
 International Algorithm Implementation Challenge:
 Problem Definitions and Specifications". Please 
 contact via e-mail: netflow@dimacs.rutgers.edu

 This program is written in standard Pascal for 
 transportability.

 Another version, that uses windowing and a user 
 friendly menu interface is available. The version
 uses Turbo Pascal 5.5 features (including CTR, DOS),
 and is compiled in Turbo Pascal Units TPU's. Runs on
 DOS 3.xx and latter versions. Five network generators
 and a new max-flow algorithm are included in the 
 package, and can be invoked from the menu.

 This program generates a BALANCED ARC TREE. This arc
 tree is temporarily maintained and manipulated in the 
 RAM. When the network is complete, i.e. the arc tree
 is filled with arcs (random pairs of numbers), the 
 network is written to a file. Two types of run-time
 errors may occur:

   101 Disk write error, if the disk becomes full
   203 Heap overflow error. With Turbo Pascal 5.5
       the stack, heap minimum and heap maximum sizes
       can be controlled using the M compiler option:
       [ M stacksize,heapmin,heapmax], with default
       being [ M 16384,0,655360]. See below an example
       use of this compiler option. 
       
       Each dynamic variable is stored in the heap by
       stacking them on the top of each other.

 In Turbo Pascal 5.5 the available memory can be tested
 using MemAvail and MaxAvail functions (omitted here for
 transportability). Please note, that relatively small
 networks, a few thousand nodes and arcs, will already
 cause a heap overflow on PC's with 512 to 640kB of RAM.
}

Program RandomNetwork(input,netres,output);
{$M 16384,0,512000}

  Type
  ArcTreeType  = ^ArcRecord;
  ArcRecord    =
    RECORD
      ArcNumber:integer;
      ii :integer;
      jj :integer;
      cap:integer;
      LeftArcTree,RightArcTree:ArcTreeType;
    End;
  type string = packed array [1..20] of char;

  Var
    Arc:ArcTreeType;
    ArcTreeRoot:ArcTreeType;
    i,num,arcnumber:integer;
    num_arcs,num_nodes,capacity:integer;
    z:integer;
    X,Y:integer;
    xx1,xx2,xx3,xx4:integer;
    source,sink:integer;
    ch:char;
    found:boolean;
    found11,found22:boolean;
    OutFileName:string;
    netres:text;

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

    Function ArcTree(nn:integer):ArcTreeType;
    Var NewArc:ArcTreeType;
        nll,nrr:integer;
  Begin
    IF nn=0 THEN ArcTree:=nil ELSE
    Begin
      nll:=nn DIV 2; nrr:=nn-nll-1;
      xx1:=xx1+1;
      xx2:=0;xx3:=0;xx4:=0;
      NEW(NewArc);
      WITH NewArc^ DO
      Begin
        ArcNumber:=xx1;
        ii:=xx2; jj:=xx3; Cap:=xx4;
        LeftArcTree :=ArcTree(nll);
        RightArcTree:=ArcTree(nrr);
      End;
      ArcTree:=NewArc;
    End;
  End;

  Procedure LocateArc(t2:ArcTreeType; h2:integer);
    VAR Found2:BOOLEAN;
    Begin
      Found2:=FALSE;
      WHILE (t2<>nil) AND NOT Found2 DO
      Begin
        IF t2^.ArcNumber=h2 THEN Found2:=TRUE
        ELSE IF h2=t2^.LeftArcTree^.ArcNumber
        THEN t2:=t2^.LeftArcTree
        ELSE IF h2=t2^.RightArcTree^.ArcNumber
        THEN t2:=t2^.RightArcTree
        ELSE IF h2<t2^.RightArcTree^.ArcNumber
        THEN t2:=t2^.LeftArcTree
        ELSE IF h2>t2^.RightArcTree^.ArcNumber
        THEN t2:=t2^.RightArcTree;
      End;
      IF Found2 THEN Arc:=t2;
    End;

  Procedure PrintArcTree(ArcTree1:ArcTreeType);
    Begin
      IF ArcTree1<>nil THEN
      WITH ArcTree1^ DO
      Begin
        PrintArcTree(LeftArcTree);
        write(netres,'a');
        writeln(netres,ii:10,jj:10,Cap:10);
        z:=z+1;
        PrintArcTree(RightArcTree);
      End;
    End;

Procedure Randomnet;
  var complete:boolean;
  Procedure Add_Arc1;
    Begin
      z:=1;
      repeat
      Begin
        LocateArc(ArcTreeRoot,Z);
        IF (Arc^.ii=0) and (Arc^.jj=0) THEN
          Begin
            Arc^.ii:=X; Arc^.jj:=Y;
            Arc^.cap:=irandom(capacity)+1;
            IF z=num_arcs THEN complete:=true;
            z:=num_arcs;
          End;
        z:=z+1;
      End;
      until z=num_arcs+1;
    End;

  Procedure Add_Arc2;
    Begin
      z:=1;
      repeat
      Begin
        LocateArc(ArcTreeRoot,Z);
        IF (Arc^.ii=0) and (Arc^.jj=0) THEN
          Begin
            Arc^.ii:=Y; Arc^.jj:=X;
            Arc^.cap:=irandom(capacity)+1;
            IF z=num_arcs THEN complete:=true;
            z:=num_arcs;
          End;
        z:=z+1;
      End;
      until z=num_arcs+1;
    End;

  Begin
    complete:=false;
    repeat
    Begin
      Begin
        repeat
          X:=irandom(num_nodes)+1;
          Y:=irandom(num_nodes)+1;
        until (Y<>X);
      End;
      IF X<>Y THEN
      Begin
        found11:=false;
        found22:=false;
        for z:=1 to num_arcs do
        Begin
          LocateArc(ArcTreeRoot,Z);
          IF ((X=Arc^.ii) and (Y=Arc^.jj))
            THEN found11:=true;
        End;
        IF not found11 THEN Add_Arc1
        else if found11 then
        Begin
          for z:=1 to num_arcs do
          Begin
            LocateArc(ArcTreeRoot,Z);
            IF ((Y=Arc^.ii) and (X=Arc^.jj))
              THEN found22:=true;
          End;
          IF not found22 THEN Add_Arc2;
        End;
      End;
    End;
    until complete;
  End;
  
  Procedure Banner1;
    Begin
      writeln;
      writeln('     RANDOM NETWORK GENERATOR');
      writeln('          FOR MAX-FLOW');
      writeln('    Copyright (C) Gary R. Waissi');
      writeln('              (1990)');
      writeln('   University of Michigan-Dearborn');
      writeln('     School of Management, 113FOB');
      writeln('        Dearborn, MI 48128');
      writeln;
      writeln('  This program generates a RANDOM');
      writeln('  network for MAX-FLOW into a user');
      writeln('  file. The network is generated by');
      writeln('  randomly pairing a given number');
      writeln('  of nodes using a given number of');
      writeln('  arcs.');
      writeln;
    End;

  Procedure Banner2;
    Begin
      writeln;
      writeln('  The maximum number of arcs in');
      writeln('  a fully dense network cannot ');
      writeln('  exceed n^2-n. Also, you need at');
      writeln('  least one arc. Duplicate arcs');
      writeln('  and self-loops will not ');
      writeln('  be allowed.    ');
    End;

  Procedure Banner3;
    Begin
      writeln;
      writeln('  A RANDOM NETWORK for MAX-FLOW will');
      writeln('  be generated to file ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The sink node is node   ',sink:8);
      writeln('  The sink node is numbered with the');
      writeln('  largest node number. ');
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
      writeln;
      writeln('  The network generation may take');
      writeln('  some time. The entire network is');
      writeln('  temporarily stored in a tree in RAM.');
      writeln('  The storage of the tree requires');
      writeln('  storing of dynamic variables in ');
      writeln('  a heap. The heap sizes are limited,');
      writeln('  but can usually be controlled using');
      writeln('  various compiler options. ');
      writeln('  PLEASE WAIT !!');
      writeln;
    End;

  Procedure Banner4;
    Begin
      writeln;
      writeln('  The network is completed. The data');
      writeln('  file is an ASCII file, with name:');
	  writeln('  ',OutFileName,',and can be');
      writeln('  accessed with your editor.');
   End;

  Procedure UserValues;
    var total_arcs:integer;
    Begin
      xx1:=0;
      repeat
        write('  How many nodes are in the network: ');
        readln(num_nodes);
        if (num_nodes<=1) then
        writeln('  Try again. Need at least two nodes.');
      until (num_nodes>1);
      total_arcs:=num_nodes*num_nodes-num_nodes;
      write('  How many arcs are in the network: ');
      Begin
        repeat
          readln(num_arcs);
          if (num_arcs>total_arcs) OR (num_arcs<1) then
            Begin
              Banner2;
              writeln('  Try again with the number of');
              writeln('  arcs at least one, but not ');
              writeln('  more than ',total_arcs,'.');
              write('  How many arcs are in the network: ');
            End;
          until ((num_arcs<=total_arcs) AND (num_arcs>=1));
      End;      
      writeln('  What is the upper bound for arc flow');
      write('  CAPACITY: ');
      readln(capacity);
      source:=1;
      sink:=num_nodes;
      Banner3;
      writeln(netres,'c Random Network');
      writeln(netres,'c for Max-Flow');
      writeln(netres,'p max',num_nodes:10,num_arcs:10);
      writeln(netres,'n',source:14,'  s');
      writeln(netres,'n',sink:14,'  t');
      ArcTreeRoot:=ArcTree(num_arcs);
    End;

  Begin
    randomize;  {Turbo Pascal random number generator initiator.}
                {You may to replace this by compiler specific   }
                {randomizer, or write a short procedure.        }
    Banner1;
    writeln('  Try the network generation out by');
    writeln('  following the instructions.');
    write('  Do you want to continue (Y/N) ');
    readln(ch);
    if (ch='y') or (ch='Y') then
    Begin
      write('  Enter name of the output file: ');
      readln(OutFileName);
{ SUN       assign(netres,OutFileName+'.max'); }
      rewrite(netres,OutFileName);
      UserValues;
      RandomNet;
      z:=1;
      PrintArcTree(ArcTreeRoot);
{ SUN       close(netres); }
      Banner4;
    End;
  End.
SHAR_EOF
cat << 'SHAR_EOF' > waissi/ra-max.pas
{
 RANDOM NETWORK GENERATOR FOR MAX-FLOW  (revised 11/25/90)

 Copyright (C) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128

 Tel: (313) 593-5012
 E-mail: gary_waissi@um.cc.mich.edu

 Revisions: INTEGER changed to LONGINT to allow the
 range of [-2^31 ... 2^31-1].

 This program generates RANDOM NETWORKS for MAX-FLOW
 applications to user specified files. Key procedure is:

 PROCEDURE RandomNet;

 Data file format corresponds to DIMACS data file
 specifications described in "The First DIMACS 
 International Algorithm Implementation Challenge:
 Problem Definitions and Specifications". Please 
 contact via e-mail: netflow@dimacs.rutgers.edu

 This program is written in standard Pascal for 
 transportability.

 Another version, that uses windowing and a user 
 friendly menu interface is available. The version
 uses Turbo Pascal 5.5 features (including CTR, DOS),
 and is compiled in Turbo Pascal Units TPU's. Runs on
 DOS 3.xx and latter versions. Five network generators
 and a new max-flow algorithm are included in the 
 package, and can be invoked from the menu.

 This program generates a BALANCED ARC TREE. This arc
 tree is temporarily maintained and manipulated in the 
 RAM. When the network is complete, i.e. the arc tree
 is filled with arcs (random pairs of numbers), the 
 network is written to a file. Two types of run-time
 errors may occur:

   101 Disk write error, if the disk becomes full
   203 Heap overflow error. With Turbo Pascal 5.5
       the stack, heap minimum and heap maximum sizes
       can be controlled using the M compiler option:
       [ M stacksize,heapmin,heapmax], with default
       being [ M 16384,0,655360]. See below an example
       use of this compiler option. 
       
       Each dynamic variable is stored in the heap by
       stacking them on the top of each other.

 In Turbo Pascal 5.5 the available memory can be tested
 using MemAvail and MaxAvail functions (omitted here for
 transportability). Please note, that relatively small
 networks, a few thousand nodes and arcs, will already
 cause a heap overflow on PC's with 512 to 640kB of RAM.
}

Program RandomNetwork(input,netres);
{$M 16384,0,512000}

  Type
  ArcTreeType  = ^ArcRecord;
  ArcRecord    =
    RECORD
      ArcNumber:longint;
      ii :longint;
      jj :longint;
      cap:longint;
      LeftArcTree,RightArcTree:ArcTreeType;
    End;

  Var
    Arc:ArcTreeType;
    ArcTreeRoot:ArcTreeType;
    i,num,arcnumber:longint;
    num_arcs,num_nodes,capacity:longint;
    z:longint;
    X,Y:longint;
    xx1,xx2,xx3,xx4:longint;
    source,sink:longint;
    ch:char;
    found:boolean;
    found11,found22:boolean;
    OutFileName:string[20];
    netres:text;

  Function ArcTree(nn:longint):ArcTreeType;
    Var NewArc:ArcTreeType;
        nll,nrr:longint;
  Begin
    IF nn=0 THEN ArcTree:=nil ELSE
    Begin
      nll:=nn DIV 2; nrr:=nn-nll-1;
      xx1:=xx1+1;
      xx2:=0;xx3:=0;xx4:=0;
      NEW(NewArc);
      WITH NewArc^ DO
      Begin
        ArcNumber:=xx1;
        ii:=xx2; jj:=xx3; Cap:=xx4;
        LeftArcTree :=ArcTree(nll);
        RightArcTree:=ArcTree(nrr);
      End;
      ArcTree:=NewArc;
    End;
  End;

  Procedure LocateArc(t2:ArcTreeType; h2:longint);
    VAR Found2:BOOLEAN;
    Begin
      Found2:=FALSE;
      WHILE (t2<>nil) AND NOT Found2 DO
      Begin
        IF t2^.ArcNumber=h2 THEN Found2:=TRUE
        ELSE IF h2=t2^.LeftArcTree^.ArcNumber
        THEN t2:=t2^.LeftArcTree
        ELSE IF h2=t2^.RightArcTree^.ArcNumber
        THEN t2:=t2^.RightArcTree
        ELSE IF h2<t2^.RightArcTree^.ArcNumber
        THEN t2:=t2^.LeftArcTree
        ELSE IF h2>t2^.RightArcTree^.ArcNumber
        THEN t2:=t2^.RightArcTree;
      End;
      IF Found2 THEN Arc:=t2;
    End;

  Procedure PrintArcTree(ArcTree1:ArcTreeType);
    Begin
      IF ArcTree1<>nil THEN
      WITH ArcTree1^ DO
      Begin
        PrintArcTree(LeftArcTree);
        write(netres,'a');
        writeln(netres,ii:10,jj:10,Cap:10);
        z:=z+1;
        PrintArcTree(RightArcTree);
      End;
    End;

Procedure Randomnet;
  var complete:boolean;
  Procedure Add_Arc1;
    Begin
      z:=1;
      repeat
      Begin
        LocateArc(ArcTreeRoot,Z);
        IF (Arc^.ii=0) and (Arc^.jj=0) THEN
          Begin
            Arc^.ii:=X; Arc^.jj:=Y;
            Arc^.cap:=random(capacity)+1;
            IF z=num_arcs THEN complete:=true;
            z:=num_arcs;
          End;
        z:=z+1;
      End;
      until z=num_arcs+1;
    End;

  Procedure Add_Arc2;
    Begin
      z:=1;
      repeat
      Begin
        LocateArc(ArcTreeRoot,Z);
        IF (Arc^.ii=0) and (Arc^.jj=0) THEN
          Begin
            Arc^.ii:=Y; Arc^.jj:=X;
            Arc^.cap:=random(capacity)+1;
            IF z=num_arcs THEN complete:=true;
            z:=num_arcs;
          End;
        z:=z+1;
      End;
      until z=num_arcs+1;
    End;

  Begin
    complete:=false;
    repeat
    Begin
      Begin
        repeat
          X:=random(num_nodes)+1;
          Y:=random(num_nodes)+1;
        until (Y<>X);
      End;
      IF X<>Y THEN
      Begin
        found11:=false;
        found22:=false;
        for z:=1 to num_arcs do
        Begin
          LocateArc(ArcTreeRoot,Z);
          IF ((X=Arc^.ii) and (Y=Arc^.jj))
            THEN found11:=true;
        End;
        IF not found11 THEN Add_Arc1
        else if found11 then
        Begin
          for z:=1 to num_arcs do
          Begin
            LocateArc(ArcTreeRoot,Z);
            IF ((Y=Arc^.ii) and (X=Arc^.jj))
              THEN found22:=true;
          End;
          IF not found22 THEN Add_Arc2;
        End;
      End;
    End;
    until complete;
  End;
  
  Procedure Banner1;
    Begin
      writeln;
      writeln('     RANDOM NETWORK GENERATOR');
      writeln('          FOR MAX-FLOW');
      writeln('    Copyright (C) Gary R. Waissi');
      writeln('              (1990)');
      writeln('   University of Michigan-Dearborn');
      writeln('     School of Management, 113FOB');
      writeln('        Dearborn, MI 48128');
      writeln;
      writeln('  This program generates a RANDOM');
      writeln('  network for MAX-FLOW into a user');
      writeln('  file. The network is generated by');
      writeln('  randomly pairing a given number');
      writeln('  of nodes using a given number of');
      writeln('  arcs.');
      writeln;
    End;

  Procedure Banner2;
    Begin
      writeln;
      writeln('  The maximum number of arcs in');
      writeln('  a fully dense network cannot ');
      writeln('  exceed n^2-n. Also, you need at');
      writeln('  least one arc. Duplicate arcs');
      writeln('  and self-loops will not ');
      writeln('  be allowed.    ');
    End;

  Procedure Banner3;
    Begin
      writeln;
      writeln('  A RANDOM NETWORK for MAX-FLOW will');
      writeln('  be generated to file ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The sink node is node   ',sink:8);
      writeln('  The sink node is numbered with the');
      writeln('  largest node number. ');
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
      writeln;
      writeln('  The network generation may take');
      writeln('  some time. The entire network is');
      writeln('  temporarily stored in a tree in RAM.');
      writeln('  The storage of the tree requires');
      writeln('  storing of dynamic variables in ');
      writeln('  a heap. The heap sizes are limited,');
      writeln('  but can usually be controlled using');
      writeln('  various compiler options. ');
      writeln('  PLEASE WAIT !!');
      writeln;
    End;

  Procedure Banner4;
    Begin
      writeln;
      writeln('  The network is completed. The data');
      writeln('  file is an ASCII file, and can be');
      writeln('  accessed with your editor. You may');
      writeln('  print the file using the DOS');
      writeln('  command PRINT ',OutFileName,'.max.');
    End;

  Procedure UserValues;
    var total_arcs:longint;
    Begin
      xx1:=0;
      repeat
        write('  How many nodes are in the network: ');
        readln(num_nodes);
        if (num_nodes<=1) then
        writeln('  Try again. Need at least two nodes.');
      until (num_nodes>1);
      total_arcs:=num_nodes*num_nodes-num_nodes;
      write('  How many arcs are in the network: ');
      Begin
        repeat
          readln(num_arcs);
          if (num_arcs>total_arcs) OR (num_arcs<1) then
            Begin
              Banner2;
              writeln('  Try again with the number of');
              writeln('  arcs at least one, but not ');
              writeln('  more than ',total_arcs,'.');
              write('  How many arcs are in the network: ');
            End;
          until ((num_arcs<=total_arcs) AND (num_arcs>=1));
      End;      
      writeln('  What is the upper bound for arc flow');
      write('  CAPACITY: ');
      readln(capacity);
      source:=1;
      sink:=num_nodes;
      Banner3;
      writeln(netres,'c Random Network');
      writeln(netres,'c for Max-Flow');
      writeln(netres,'p max',num_nodes:10,num_arcs:10);
      writeln(netres,'n',source:14,'  s');
      writeln(netres,'n',sink:14,'  t');
      ArcTreeRoot:=ArcTree(num_arcs);
    End;

  Begin
    randomize;  {Turbo Pascal random number generator initiator.}
                {You may to replace this by compiler specific   }
                {randomizer, or write a short procedure.        }
    Banner1;
    writeln('  Try the network generation out by');
    writeln('  following the instructions.');
    write('  Do you want to continue (Y/N) ');
    readln(ch);
    if (ch='y') or (ch='Y') then
    Begin
      write('  Enter name of the output file: ');
      readln(OutFileName);
      assign(netres,OutFileName+'.max');
      rewrite(netres);
      UserValues;
      RandomNet;
      z:=1;
      PrintArcTree(ArcTreeRoot);
      close(netres);
      Banner4;
    End;
  End.
SHAR_EOF
cat << 'SHAR_EOF' > waissi/random.p
{ 
  MODIFIED TO COMPILE UNDER SUN UNIX PASCAL 11/90 C. McGeoch
  Modifications include:
    1) Change all integer types ot integer types
    2) Define a string type
    3) Change assign to rewrite
    4) Remove the close function
    5) Add function randomize;
    6) Add function irandom;
    7) Remove user instructions that no longer apply
  
  Catherine McGeoch
  DIMACS Center 
  Box 1179, Rutgers University
  Piscataway, NJ 08855-1179
 }

{
 RANDOM NETWORK GENERATOR FOR MAX-FLOW  (revised 11/25/90)
 
 Copyright (C) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128
 
 Tel: (313) 593-5012
 E-mail: gary_waissi@um.cc.mich.edu
 
 Revisions: INTEGER changed to LONGINT to allow the
 range of [-2^31 ... 2^31-1].
 
 This program generates RANDOM NETWORKS to user
 specified files. This program was originally 
 designed for max-flow applications. Key procedure
 is:
 
 PROCEDURE RandomNet;
 
 Data file format corresponds to DIMACS data file
 specifications described in "The First DIMACS 
 International Algorithm Implementation Challenge:
 Problem Definitions and Specifications". Please 
 contact via e-mail: netflow@dimacs.rutgers.edu
 
 This program is written in standard Pascal for 
 transportability.
 
 Another version, that uses windowing and a user 
 friendly menu interface is available. The version
 uses Turbo Pascal 5.5 features (including CTR, DOS),
 and is compiled in Turbo Pascal Units TPU's. Runs on
 DOS 3.xx and latter versions. Five network generators
 and a new max-flow algorithm are included in the 
 package, and can be invoked from the menu.
 
 This program generates a BALANCED ARC TREE. This arc
 tree is temporarily maintained and manipulated in the 
 RAM. When the network is complete, i.e. the arc tree
 is filled with arcs (random pairs of numbers), the 
 network is written to a file. Two types of run-time
 errors may occur:
 
   101 Disk write error, if the disk becomes full
   203 Heap overflow error. With Turbo Pascal 5.5
       the stack, heap minimum and heap maximum sizes
       can be controlled using the M compiler option:
       [ M stacksize,heapmin,heapmax], with default
       being [ M 16384,0,655360]. See below an example
       use of this compiler option. 
 
       Each dynamic variable is stored in the heap by
       stacking them on the top of each other.
 
 In Turbo Pascal 5.5 the available memory can be tested
 using MemAvail and MaxAvail functions (omitted here for
 transportability). Please note, that relatively small
 networks, a few thousand nodes and arcs, will already
 cause a heap overflow on PC's with 512 to 640kB of RAM.
}
 
Program RandomNetwork(input,netres,output);
{$M 16384,0,512000}
  Type
  ArcTreeType  = ^ArcRecord;
  ArcRecord    =
    RECORD
      ArcNumber:integer;
      ii :integer;
      jj :integer;
      cap:integer;
      LeftArcTree,RightArcTree:ArcTreeType;
    End;
 string = packed array [1..20] of char;
    
  Var
    Arc:ArcTreeType;
    ArcTreeRoot:ArcTreeType;
    i,k,num,arcnumber:integer;
    num_arcs,num_nodes,capacity:integer;
    z:integer;
    X,Y:integer;
    xx1,xx2,xx3,xx4:integer;
    source,sink:integer;
    ch:char;
    found,example:boolean;
    found11,found22:boolean;
    OutFileName:string;
    netres:text;

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
 
  Function ArcTree(nn:integer):ArcTreeType;
    Var NewArc:ArcTreeType;
        nll,nrr:integer;
  Begin
    IF nn=0 THEN ArcTree:=nil ELSE
    Begin
      nll:=nn DIV 2; nrr:=nn-nll-1;
      xx1:=xx1+1;
      xx2:=0;xx3:=0;xx4:=0;
      NEW(NewArc);
      WITH NewArc^ DO
      Begin
        ArcNumber:=xx1;
        ii:=xx2; jj:=xx3; Cap:=xx4;
        LeftArcTree :=ArcTree(nll);
        RightArcTree:=ArcTree(nrr);
      End;
      ArcTree:=NewArc;
    End;
  End;
 
  Procedure LocateArc(t2:ArcTreeType; h2:integer);
    VAR Found2:BOOLEAN;
    Begin
      Found2:=FALSE;
      WHILE (t2<>nil) AND NOT Found2 DO
      Begin
        IF t2^.ArcNumber=h2 THEN Found2:=TRUE
        ELSE IF h2=t2^.LeftArcTree^.ArcNumber
        THEN t2:=t2^.LeftArcTree
        ELSE IF h2=t2^.RightArcTree^.ArcNumber
        THEN t2:=t2^.RightArcTree
        ELSE IF h2<t2^.RightArcTree^.ArcNumber
        THEN t2:=t2^.LeftArcTree
        ELSE IF h2>t2^.RightArcTree^.ArcNumber
        THEN t2:=t2^.RightArcTree;
      End;
      IF Found2 THEN Arc:=t2;
    End;
 
  Procedure PrintArcTree(ArcTree1:ArcTreeType);
    Begin
      IF ArcTree1<>nil THEN
      WITH ArcTree1^ DO
      Begin
        PrintArcTree(LeftArcTree);
        write(netres,'a');
        writeln(netres,ii:8,jj:8,Cap:8);
        if k<5 then Writeln(ii:5,jj:5,Cap:5);
        k:=k+1;
        z:=z+1;
        PrintArcTree(RightArcTree);
      End;
    End;
 
Procedure Randomnet;
  var complete:boolean;
      iter1,iter2:boolean;
  Procedure Add_Arc1;
    Begin
      z:=1;
      repeat
      Begin
        LocateArc(ArcTreeRoot,Z);
        IF (Arc^.ii=0) and (Arc^.jj=0) THEN
          Begin
            Arc^.ii:=X; Arc^.jj:=Y;
            Arc^.cap:=irandom(capacity)+1;
            IF z=num_arcs THEN complete:=true;
            z:=num_arcs;
          End;
        z:=z+1;
      End;
      until z=num_arcs+1;
    End;
 
  Procedure Add_Arc2;
    Begin
      z:=1;
      repeat
      Begin
        LocateArc(ArcTreeRoot,Z);
        IF (Arc^.ii=0) and (Arc^.jj=0) THEN
          Begin
            Arc^.ii:=Y; Arc^.jj:=X;
            Arc^.cap:=irandom(capacity)+1;
            IF z=num_arcs THEN complete:=true;
            z:=num_arcs;
          End;
        z:=z+1;
      End;
      until z=num_arcs+1;
    End;
 
  Begin
    iter1:=true;
    iter2:=true;
    complete:=false;
    repeat
    Begin
      Begin
        repeat
          if iter1 and iter2 then
          begin
            X:=irandom(num_nodes)+1; Y:=irandom(num_nodes)+1;
            iter2:=false;
          end
          else if iter1 and not iter2 then
          begin
            Y:=irandom(num_nodes)+1; X:=irandom(num_nodes)+1;
            iter1:=false;
            iter2:=true;
          end
          else if not iter1 and iter2 then
          begin
            X:=irandom(X)+1; Y:=irandom(Y)+1;
            iter2:=false;
          end
          else if not iter1 and not iter2 then
          begin
            Y:=irandom(X)+1; X:=irandom(Y)+1;
            iter1:=true;
            iter2:=true;
          end;
        until (Y<>X);
      End;
      IF X<>Y THEN
      Begin
        found11:=false;
        found22:=false;
        for z:=1 to num_arcs do
        Begin
          LocateArc(ArcTreeRoot,Z);
          IF ((X=Arc^.ii) and (Y=Arc^.jj)) THEN found11:=true;
        End;
        IF not found11 THEN Add_Arc1
        else if found11 then
        Begin
          for z:=1 to num_arcs do
          Begin
            LocateArc(ArcTreeRoot,Z);
            IF ((Y=Arc^.ii) and (X=Arc^.jj)) THEN found22:=true;
          End;
          IF not found22 THEN Add_Arc2;
        End;
      End;
    End;
    until complete;
  End;
 
  Procedure UserValues;
    var total_arcs:integer;
    Begin
      xx1:=0;
      writeln('  RANDOM NETWORK GENERATION');
      repeat
        write('  How many nodes are in the network: ');
        readln(num_nodes);
        if (num_nodes<=1) then
        writeln('  Try again. Need at least two nodes.');
      until (num_nodes>1);
      total_arcs:=num_nodes*num_nodes-num_nodes;
      write('  How many arcs are in the network: ');
      Begin
        repeat
          readln(num_arcs);
          if (num_arcs>total_arcs) OR (num_arcs<1) then
            Begin
              writeln('  The maximum number of arcs in a fully ');
              writeln('  dense network cannot exceed n^2-n.    ');
              writeln('  Also, you need at least one arc.');
              writeln('  Duplicate arcs and self-loops will not ');
              writeln('  be allowed.    ');
              writeln('  Try again with the number of arcs at');
              writeln('  least one, but not more than ',total_arcs,'.');
              write('  How many arcs are in the network: ');
            End;
          until ((num_arcs<=total_arcs) AND (num_arcs>=1));
      End; 
      write('  What is the upper bound for arc flow capacity: ');
      readln(capacity);
      source:=1;
      sink:=num_nodes;
      writeln(netres,'c Random Network');
      writeln(netres,'p max',num_nodes:6,num_arcs:8);
      writeln(netres,'n',source:10,'  s');
      writeln(netres,'n',sink:10,'  t');
      writeln;
      writeln('  A RANDOM NETWORK  will be generated to file ');
      writeln('  ',OutFileName,'.');
      writeln('  The number of nodes is ',num_nodes:8);
      writeln('  The number of arcs is  ',num_arcs:8);
      writeln('  The source node is node ',source,'. The sink node');
      writeln('  is numbered with the largest node number. Here the');
      writeln('  sink is node ',sink,'. '); 

{*SUN The network is given as  ');
      writeln('  a list of arcs using four numbers as follows:');
      writeln('  arc tail node - arc head node - arc cost/capacity.');
      writeln('  The first few data lines look like this:');
      writeln;
      writeln('   Number of');
      writeln('   Nodes     Arcs');
      writeln(num_nodes:5,num_arcs:10);
      writeln('   Source    Sink');
      writeln(source:5,sink:10);
      writeln('   From  To  Capacity');
}

      writeln('   PLEASE WAIT !!!!      ');
      ArcTreeRoot:=ArcTree(num_arcs);
    End;
 
  Begin
    randomize;  {Turbo Pascal random number generator initiator.}
                {You may to replace this by compiler specific   }
                {randomizer, or write a short procedure.        }
    example:=true;
    writeln('  This program generates a RANDOM network');
    writeln('  into a user file. The network is generated');
    writeln('  by randomly pairing a given number of nodes');
    writeln('  using a given number of arcs.');
    writeln;
    writeln('  Try the network generation out by following the');
    writeln('  instructions.');
    write('  Do you want to continue (Y/N) ');
    readln(ch);
    if (ch='y') or (ch='Y') then
    Begin
      write('  Enter name of the output file: ');
      readln(OutFileName);

{*SUN       assign(netres,OutFileName+'.max');
}
      rewrite(netres,OutFileName);
      UserValues;
      RandomNet;
      z:=1; k:=1;
      PrintArcTree(ArcTreeRoot);
{*SUN      close(netres);
}
      writeln;
      writeln('  The network is completed. The data file is');
      writeln('  an ASCII file, and can be accessed with your');
      writeln('  editor.');
    End;
  End.
SHAR_EOF
cat << 'SHAR_EOF' > waissi/random.pas
Received: from porthos.rutgers.edu by dimacs.rutgers.edu (5.59/SMI4.0/RU1.4/3.08) 
	id AA13664; Mon, 26 Nov 90 01:02:11 EST
Received: from umich.edu by porthos.rutgers.edu (5.59/SMI4.0/RU1.4/3.08) 
	id AA10289; Sun, 25 Nov 90 21:19:57 EST
Received: from um.cc.umich.edu by umich.edu (5.61/1123-1.0)
	id AA10892; Sun, 25 Nov 90 21:19:53 -0500
Date: Sun, 25 Nov 90 21:16:26 EST
From: Gary_Waissi@um.cc.umich.edu
To: netflow@dimacs.rutgers.edu
Message-Id: <7319988@um.cc.umich.edu>
Status: RO

{
 RANDOM NETWORK GENERATOR FOR MAX-FLOW  (revised 11/25/90)
 
 Copyright (C) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128
 
 Tel: (313) 593-5012
 E-mail: gary_waissi@um.cc.mich.edu
 
 Revisions: INTEGER changed to LONGINT to allow the
 range of [-2^31 ... 2^31-1].
 
 This program generates RANDOM NETWORKS to user
 specified files. This program was originally 
 designed for max-flow applications. Key procedure
 is:
 
 PROCEDURE RandomNet;
 
 Data file format corresponds to DIMACS data file
 specifications described in "The First DIMACS 
 International Algorithm Implementation Challenge:
 Problem Definitions and Specifications". Please 
 contact via e-mail: netflow@dimacs.rutgers.edu
 
 This program is written in standard Pascal for 
 transportability.
 
 Another version, that uses windowing and a user 
 friendly menu interface is available. The version
 uses Turbo Pascal 5.5 features (including CTR, DOS),
 and is compiled in Turbo Pascal Units TPU's. Runs on
 DOS 3.xx and latter versions. Five network generators
 and a new max-flow algorithm are included in the 
 package, and can be invoked from the menu.
 
 This program generates a BALANCED ARC TREE. This arc
 tree is temporarily maintained and manipulated in the 
 RAM. When the network is complete, i.e. the arc tree
 is filled with arcs (random pairs of numbers), the 
 network is written to a file. Two types of run-time
 errors may occur:
 
   101 Disk write error, if the disk becomes full
   203 Heap overflow error. With Turbo Pascal 5.5
       the stack, heap minimum and heap maximum sizes
       can be controlled using the M compiler option:
       [ M stacksize,heapmin,heapmax], with default
       being [ M 16384,0,655360]. See below an example
       use of this compiler option. 
 
       Each dynamic variable is stored in the heap by
       stacking them on the top of each other.
 
 In Turbo Pascal 5.5 the available memory can be tested
 using MemAvail and MaxAvail functions (omitted here for
 transportability). Please note, that relatively small
 networks, a few thousand nodes and arcs, will already
 cause a heap overflow on PC's with 512 to 640kB of RAM.
}
 
Program RandomNetwork(input,netres);
{$M 16384,0,512000}
 
  Type
  ArcTreeType  = ^ArcRecord;
  ArcRecord    =
    RECORD
      ArcNumber:longint;
      ii :longint;
      jj :longint;
      cap:longint;
      LeftArcTree,RightArcTree:ArcTreeType;
    End;
 
  Var
    Arc:ArcTreeType;
    ArcTreeRoot:ArcTreeType;
    i,k,num,arcnumber:longint;
    num_arcs,num_nodes,capacity:longint;
    z:longint;
    X,Y:longint;
    xx1,xx2,xx3,xx4:longint;
    source,sink:longint;
    ch:char;
    found,example:boolean;
    found11,found22:boolean;
    OutFileName:string[20];
    netres:text;
 
  Function ArcTree(nn:longint):ArcTreeType;
    Var NewArc:ArcTreeType;
        nll,nrr:longint;
  Begin
    IF nn=0 THEN ArcTree:=nil ELSE
    Begin
      nll:=nn DIV 2; nrr:=nn-nll-1;
      xx1:=xx1+1;
      xx2:=0;xx3:=0;xx4:=0;
      NEW(NewArc);
      WITH NewArc^ DO
      Begin
        ArcNumber:=xx1;
        ii:=xx2; jj:=xx3; Cap:=xx4;
        LeftArcTree :=ArcTree(nll);
        RightArcTree:=ArcTree(nrr);
      End;
      ArcTree:=NewArc;
    End;
  End;
 
  Procedure LocateArc(t2:ArcTreeType; h2:longint);
    VAR Found2:BOOLEAN;
    Begin
      Found2:=FALSE;
      WHILE (t2<>nil) AND NOT Found2 DO
      Begin
        IF t2^.ArcNumber=h2 THEN Found2:=TRUE
        ELSE IF h2=t2^.LeftArcTree^.ArcNumber
        THEN t2:=t2^.LeftArcTree
        ELSE IF h2=t2^.RightArcTree^.ArcNumber
        THEN t2:=t2^.RightArcTree
        ELSE IF h2<t2^.RightArcTree^.ArcNumber
        THEN t2:=t2^.LeftArcTree
        ELSE IF h2>t2^.RightArcTree^.ArcNumber
        THEN t2:=t2^.RightArcTree;
      End;
      IF Found2 THEN Arc:=t2;
    End;
 
  Procedure PrintArcTree(ArcTree1:ArcTreeType);
    Begin
      IF ArcTree1<>nil THEN
      WITH ArcTree1^ DO
      Begin
        PrintArcTree(LeftArcTree);
        write(netres,'a');
        writeln(netres,ii:8,jj:8,Cap:8);
        if k<5 then Writeln(ii:5,jj:5,Cap:5);
        k:=k+1;
        z:=z+1;
        PrintArcTree(RightArcTree);
      End;
    End;
 
Procedure Randomnet;
  var complete:boolean;
      iter1,iter2:boolean;
  Procedure Add_Arc1;
    Begin
      z:=1;
      repeat
      Begin
        LocateArc(ArcTreeRoot,Z);
        IF (Arc^.ii=0) and (Arc^.jj=0) THEN
          Begin
            Arc^.ii:=X; Arc^.jj:=Y;
            Arc^.cap:=random(capacity)+1;
            IF z=num_arcs THEN complete:=true;
            z:=num_arcs;
          End;
        z:=z+1;
      End;
      until z=num_arcs+1;
    End;
 
  Procedure Add_Arc2;
    Begin
      z:=1;
      repeat
      Begin
        LocateArc(ArcTreeRoot,Z);
        IF (Arc^.ii=0) and (Arc^.jj=0) THEN
          Begin
            Arc^.ii:=Y; Arc^.jj:=X;
            Arc^.cap:=random(capacity)+1;
            IF z=num_arcs THEN complete:=true;
            z:=num_arcs;
          End;
        z:=z+1;
      End;
      until z=num_arcs+1;
    End;
 
  Begin
    iter1:=true;
    iter2:=true;
    complete:=false;
    repeat
    Begin
      Begin
        repeat
          if iter1 and iter2 then
          begin
            X:=random(num_nodes)+1; Y:=random(num_nodes)+1;
            iter2:=false;
          end
          else if iter1 and not iter2 then
          begin
            Y:=random(num_nodes)+1; X:=random(num_nodes)+1;
            iter1:=false;
            iter2:=true;
          end
          else if not iter1 and iter2 then
          begin
            X:=random(X)+1; Y:=random(Y)+1;
            iter2:=false;
          end
          else if not iter1 and not iter2 then
          begin
            Y:=random(X)+1; X:=random(Y)+1;
            iter1:=true;
            iter2:=true;
          end;
        until (Y<>X);
      End;
      IF X<>Y THEN
      Begin
        found11:=false;
        found22:=false;
        for z:=1 to num_arcs do
        Begin
          LocateArc(ArcTreeRoot,Z);
          IF ((X=Arc^.ii) and (Y=Arc^.jj)) THEN found11:=true;
        End;
        IF not found11 THEN Add_Arc1
        else if found11 then
        Begin
          for z:=1 to num_arcs do
          Begin
            LocateArc(ArcTreeRoot,Z);
            IF ((Y=Arc^.ii) and (X=Arc^.jj)) THEN found22:=true;
          End;
          IF not found22 THEN Add_Arc2;
        End;
      End;
    End;
    until complete;
  End;
 
  Procedure UserValues;
    var total_arcs:longint;
    Begin
      xx1:=0;
      writeln('  RANDOM NETWORK GENERATION');
      repeat
        write('  How many nodes are in the network: ');
        readln(num_nodes);
        if (num_nodes<=1) then
        writeln('  Try again. Need at least two nodes.');
      until (num_nodes>1);
      total_arcs:=num_nodes*num_nodes-num_nodes;
      write('  How many arcs are in the network: ');
      Begin
        repeat
          readln(num_arcs);
          if (num_arcs>total_arcs) OR (num_arcs<1) then
            Begin
              writeln('  The maximum number of arcs in a fully ');
              writeln('  dense network cannot exceed n^2-n.    ');
              writeln('  Also, you need at least one arc.');
              writeln('  Duplicate arcs and self-loops will not ');
              writeln('  be allowed.    ');
              writeln('  Try again with the number of arcs at');
              writeln('  least one, but not more than ',total_arcs,'.');
              write('  How many arcs are in the network: ');
            End;
          until ((num_arcs<=total_arcs) AND (num_arcs>=1));
      End; 
      write('  What is the upper bound for arc flow capacity: ');
      readln(capacity);
      source:=1;
      sink:=num_nodes;
      writeln(netres,'c Random Network');
      writeln(netres,'p max',num_nodes:6,num_arcs:8);
      writeln(netres,'n',source:10,'  s');
      writeln(netres,'n',sink:10,'  t');
      writeln;
      writeln('  A RANDOM NETWORK  will be generated to file ');
      writeln('  ',OutFileName,'.');
      writeln('  The number of nodes is ',num_nodes:8);
      writeln('  The number of arcs is  ',num_arcs:8);
      writeln('  The source node is node ',source,'. The sink node');
      writeln('  is numbered with the largest node number. Here the');
      writeln('  sink is node ',sink,'. The network is given as  ');
      writeln('  a list of arcs using four numbers as follows:');
      writeln('  arc tail node - arc head node - arc cost/capacity.');
      writeln('  The first few data lines look like this:');
      writeln;
      writeln('   Number of');
      writeln('   Nodes     Arcs');
      writeln(num_nodes:5,num_arcs:10);
      writeln('   Source    Sink');
      writeln(source:5,sink:10);
      writeln('   From  To  Capacity');
      writeln('   PLEASE WAIT !!!!      ');
      ArcTreeRoot:=ArcTree(num_arcs);
    End;
 
  Begin
    randomize;  {Turbo Pascal random number generator initiator.}
                {You may to replace this by compiler specific   }
                {randomizer, or write a short procedure.        }
    example:=true;
    writeln('  This program generates a RANDOM network');
    writeln('  into a user file. The network is generated');
    writeln('  by randomly pairing a given number of nodes');
    writeln('  using a given number of arcs.');
    writeln;
    writeln('  Try the network generation out by following the');
    writeln('  instructions.');
    write('  Do you want to continue (Y/N) ');
    readln(ch);
    if (ch='y') or (ch='Y') then
    Begin
      write('  Enter name of the output file: ');
      readln(OutFileName);
      assign(netres,OutFileName+'.max');
      rewrite(netres);
      UserValues;
      RandomNet;
      z:=1; k:=1;
      PrintArcTree(ArcTreeRoot);
      close(netres);
      writeln;
      writeln('  The network is completed. The data file is');
      writeln('  an ASCII file, and can be accessed with your');
      writeln('  editor. You may print the file using the DOS');
      writeln('  command PRINT ',OutFileName,'.max.');
    End;
  End.
SHAR_EOF
cat << 'SHAR_EOF' > waissi/readme
-----------------------------------------------------------
NETWORK PROBLEM GENERATORS

          readme_next describes five network generators which generate
          networks for maximum-flow problems in the DIMACS standard format:
		
           ac-max.p(as)    Acyclic network generator
           bi-max.p(as)    Bipartite network generator
           tr1-max.p(as)   Transit grid network generator, one-way
           tr2-max.p(as)   Transit grid network generator, two-way
           ra-max.p(as)    Random network generator

         (C) Gary R. Waissi, University of Michigan-Dearborn
         School of Management, Room 113FOB, Dearborn, MI 48128
         Tel: (313) 593-5012
         E-mail: gary_waissi@um.cc.umich.edu

-----------------------------------------------------------
DIMACS NOTE: 
     1) All file names in the directory are  in lower-case.
     2) Files with the suffix .pas will compile under Standard Pascal
     3) Files with the suffix .p   have been modified to compile
              under Sun Pascal.  You must use the -L option
              when compiling:  for example  pc -L random.p
     4) Files in the tp55 directory will compile under Turbo-Pascal 5.5	


SHAR_EOF
cat << 'SHAR_EOF' > waissi/readme_next
Gary R. Waissi, University of Michigan-Dearborn
School of Management, Room 113FOB, Dearborn, MI 48128
Tel: (313) 593-5012
E-mail: gary_waissi@um.cc.umich.edu
-----------------------------------------------------------
Table of Contents:
1. NETWORK PROBLEM GENERATORS FOR MAX-FLOW;
   Standard Pascal Versions
2. DESCRIPTION OF NETWORK GENERATORS
3. EXAMPLE OUTPUT DATA FILES
-----------------------------------------------------------
Programs:  
Copyright (C) Gary R. Waissi (1990,1991)

Algorithms:
Copyright (C) Gary R. Waissi (1988,1989,1990,1991)

(updated November, 1990, December 1990, January 1991)
-----------------------------------------------------------
1. NETWORK PROBLEM GENERATORS FOR MAX-FLOW;
   Standard Pascal Versions
-----------------------------------------------------------

The Max-Flow Problem
--------------------
In a maximum flow problem the total flow from a specified
source node to a specified sink is maximized, while 
satisfying flow feasibility and flow conservation 
constraints, as follows:
        
    MAX  SUM f(i,j) over all arcs (i,j) in A

    where f(i,j) = flow on arc (i,j)
          A      = set of arcs in the network

The Network Generators
----------------------
The programs generate max-flow networks in five network
categories; acyclic, bipartite, one-way transit grid, 
two-way transit grid, and random.

Standard Pascal Source Files
----------------------------
  ac-max.pas    Acyclic network generator
  bi-max.pas    Bipartite network generator
  tr1-max.pas   Transit grid network generator, one-way
  tr2-max.pas   Transit grid network generator, two-way
  ra-max.pas    Random network generator

These source programs have been given to public domain
through The Center for Discrete Mathematics and Theoretical
Computer Science (DIMACS), Rutgers University. The programs
are designed for maximum flow applications. All programs
are written in standard Pascal for portability.

Data file formats correspond to DIMACS data file
specifications described in "The First DIMACS International
Algorithm Implementation Challenge: Problem Definitions and
Specifications". Please contact via e-mail:
netflow@dimacs.rutgers.edu

Data File Format
----------------
All the MAX-FLOW networks are given as lists of lines using
the same format. There are four types of lines: comment 
lines c, problem line p, node lines n, arc lines a, as 
follows:

      c This is a comment line. The "p" line gives the 
      c problem type, here MAX (maximization), number of
      c nodes, and number of arcs. There are exactly two
      c "n" lines. The first "n" line lists the source node
      c number, and a lower case letter s designating the
      c source. The second "n" line lists the sink node
      c number, and a lower case c letter t designating
      c the sink. All user input must be in integers.
      c The "a" lines list the arcs, one line per arc with
      c three numbers: Arc tail node, arc head node, arc
      c flow capacity.
      c For example like this:
      p max NODES ARCS
      n NODE WHICH
      a FROM TO CAP
      c where
      c      max   = specifies the max-flow data file
      c      NODES = number of nodes
      c      ARCS  = number of arcs
      c      NODE  = source of sink node number
      c      WHICH = s for a source node, t for a sink node
      c      FROM  = arc tail node number
      c      TO    = arc head node number
      c      CAP   = arc flow capacity

For simplicity the only supply node s is always numbered
with 1, and the only demand node t is assigned the largest
node number. In all cases, where the user is asked for 
a flow capacity bound, the program will generate random
arc flow capacities. All arc flow lower bounds are set 
equal to zero.

The data files are ASCII files, and can be accessed with
a common editor. 


2. DESCRIPTION OF NETWORK GENERATORS

Acyclic Network Generator for Max-Flow
--------------------------------------
Source Files:
ac-max.pas   Standard Pascal Version
Copyright (C) Gary R. Waissi (1990,1991)

The program generates an ACYCLIC MAX-FLOW network into
a user file. An acyclic network is such that for each
arc (i,j), where i is the arc tail node number, and j is 
the arc head node number, i<j.

The program can generate three types of acyclic networks:

    - Fully dense networks with random capacities
    - Fully dense networks with special capacities
    - Special sparse networks with special capacities.

The two key procedures are:

 PROCEDURE AcyclicNet1;
 PROCEDURE AcyclicNet2;

For the FULLY DENSE acyclic networks the user has two
options for arc flow capacities: random or special
capacities. Special arcs capacities are calculated using
the concept of Glover et al. (published in "A Comprehensive
Computer Evaluation and Enhancement of Maximum Flow
Algorithms", Appl. of MS, Vol 3, 1983) as follows:

     1. For all arcs, say (i,j), where j=i+1 the capacity
        CAP(i,j) of the arc (i,j) is

            CAP(i,j) = 1 + (i - n/2)^2
            where n is the number of nodes

     2. For all arcs, say (i,j), where j>i+1 the capacity

            CAP(i,j) = 1.

The networks were called "hard networks" in the study,
because such networks were found to cause max-flow 
algorithms to their worst case performance.

If random capacities are selected for fully dense networks,
then the program wil generate random arc flow capacities
from a range of values between zero and a user specified
value. All non-integer values are truncated to integer 
values to conform with DIMACS specifications.

The special SPARSE acyclic networks are such that each node
is connected by an arc to the next node and to the sink 
node, i.e. there exist two types of arcs: 

     1. (i,j) where j=i+1 for all i and j\t with capacity n

     2. (i,t) for all i\t with capacity 1.

These simple networks cause the Dinic Layered Network
Algorithm (Dinic, E.A., Algorithm for Solution of a Problem
of Maximum Flow in a Network with Power Estimation, Soviet
Math. Dokl. Vol. 11, No. 5, pp. 1277-1280, 1970), to it's
worst case performance. That is, an acyclic network of n
nodes requires always the generation of (n-1) Dinic 
networks for maximum value flow, regardless of the maximal
flow algorithm applied to the Dinic networks. Many max-flow
algorithms use the Dinic Algorithm to generate auxiliary 
acyclic networks from an original network, and find the 
maximal flow in such an acyclic network. These sparse 
acyclic networks were presented in the dissertation of
Gary R. Waissi, "Acyclic Network Generation and Maximal
Flow Algorithms for Single Commodity Flow", University 
of Michigan, 1985.

This program is robust. A generated arc is written to the
disk file, and not stored in the memory. Disk full causes
a run-time error (101).


Bipartite Network Generator for Max-Flow
----------------------------------------
Source File:
bi-max.pas   Standard Pascal Version
Copyright (C) Gary R. Waissi (1990,1991)

The program generates a BIPARTITE MAX-FLOW NETWORK to 
a user specified file. The set of nodes N can be 
partitioned into two sets N1 and N2, such that all arcs
are directed from N1 to N2.

The program generates two types of networks:

    - Bipartite networks with unit capacities on bipartite
      arcs, and random capacities on arcs from the
      SUPER SOURCE and into the SUPER SINK.

    - Bipartite networks with random capacities.

The key procedure is:

PROCEDURE BipartiteNet;

The user selects the number of nodes on the source side
and sink side of the network respectively. The bipartite
network is appended with two nodes, a common source
SUPER SOURCE, and a common sink SUPER SINK. The SUPER
SOURCE is connected by arcs to all nodes in the SOURCE SET,
N1. The nodes in the SINK SET, N2, are connected by arcs
to the SUPER SINK. The SUPER SOURCE is always numbered with
number one, and the SUPER SINK is numbered with the largest
node number.

The user has two options for the arc capacities: either
random capacities or unit capacities. In the case of random
capacities all arcs capacities are random integers between
zero and a user selected upper bound. In the case of unit
capacities all arc capacities connecting the nodes in the
SOURCE SET, N1, to the nodes in the SINK SET, N2, are unit
capacities, i.e the bipartite arcs. However, the arc 
capacities of arcs from the SUPER SOURCE to N1, as well
as, the arc capacities of arcs from N2 to the SUPER SINK,
are random.

This program is robust. A generated arc is written to the
disk file, and not stored in the memory. Disk full causes
a run-time error (101).


Transit Grid Network Generator for Max-Flow/One-Way System
----------------------------------------------------------
Source File:
tr1-max.pas   Standard Pascal Version

Copyright (C) Gary R. Waissi (1990,1991)

This program generates a ONE-WAY TRANSIT GRID NETWORK for
the MAX-FLOW problem into a user specified file with at
most one arc between any pair of nodes.

The program generates one type of networks:
    
   - One-Way Transit-Grid networks with a SUPER SOURCE
     and a SUPER SINK, random arc flow capacities.

The key procedure is:

PROCEDURE TransitOneNet;

The network resembles a one-way city street network. The
direction of each arc, except those connected to the common
source and common sink respectively, is randomly assigned.
The arc flow capacities are randomly assigned from a range
between zero and a user selected value. All arc capacities
are random integers.

Two nodes a SUPER SOURCE and a SUPER SINK are added to the
network automatically. The SUPER SOURCE is connected by 
arcs to nodes on one side of the grid. The nodes on the 
opposite side of the grid are connected to the SUPER SINK. 

The user is suggested to select a number of nodes in the
network, that creates a complete square grid. The program
works, however, for any n>=4.

For example:

      4 nodes in a 2 x 2 network  (smallest)
      9 nodes in a 3 x 3 network
     16 nodes in a 4 x 4 network
     25 nodes in a 5 x 5 network
          ...
   1600 nodes in a 40 x 40 network
          ...
  10000 nodes in a 100 x 100 network.
          ...

This program is robust. A generated arc is written to the
disk file, and not stored in the memory. Disk full causes
a run-time error (101).

Transit Grid Network Generator for Max-Flow/Two-Way System
----------------------------------------------------------
Source File:
tr2-max.pas   Standard Pascal Version

Copyright (C) Gary R. Waissi  (1990,1991)

This program generates a TWO-WAY TRANSIT GRID NETWORK for
the MAX-FLOW problem into a user specified file with two
arcs between any pair of nodes forming the grid.

The program generates one type of networks:
    
   - Two-Way Transit-Grid networks with a SUPER SOURCE
     and a SUPER SINK, random arc flow capacities.

The key procedure is:

PROCEDURE TransitTwoNet;

The network resembles a two-way city street network. There
are two arcs, one in each direction, between the nodes 
forming the grid. The arc flow capacities are randomly 
assigned from a range between zero and a user selected 
value. 

Two nodes a SUPER SOURCE and a SUPER SINK are added to
the network automatically. The SUPER SOURCE is connected
by two arcs, one in each direction, to nodes on one side of
the grid. The nodes on the opposite side of the grid are
connected by two arcs, one in each direction, to the SUPER
SINK. 

The user is suggested to select a number of nodes in the
network, that creates a complete square grid. The program
works, however, for any n>=4.

For example:

      4 nodes in a 2 x 2 network  (smallest)
      9 nodes in a 3 x 3 network
     16 nodes in a 4 x 4 network
     25 nodes in a 5 x 5 network
          ...
   1600 nodes in a 40 x 40 network
          ...
  10000 nodes in a 100 x 100 network.
          ...

This program is robust. A generated arc is written to the
disk file, and not stored in the memory. Disk full causes
a run-time error (101).


RANDOM NETWORK GENERATOR FOR MAX-FLOW
Source File: 
ra-max.pas   Standard Pascal Version
Copyright (C) Gary R. Waissi (1990,1991)

This program generates a RANDOM MAX-FLOW network into
a user file. The network is generated by randomly pairing
a given number of nodes using a given number of arcs.

The program generates one type of networks:

    - Random networks with random arc flow capacities.

The key procedure is:

PROCEDURE RandomNet;

The maximum number of arcs in a fully dense network cannot
exceed n^2-n. Duplicate arcs and self-loops are not allowed.

This program generates initially a BALANCED ARC TREE to 
store network arcs. This arc tree is temporarily maintained
and manipulated in the RAM. The size of the tree is 
determined by the number of arcs in the random network
selected by the user. The program then generates a pair
of random numbers, say (i,j). This pair of numbers 
represents an arc (i,j) with node i as the arc origin and
node j as the arc destination. The balanced tree is then 
searched to determine, if this arc (i,j) is already 
included in the tree (network). If the pair of numbers
is found, another pair is generated and the search is 
repeated. The process is repeated until the tree (network)
is complete, i.e. the required number of arcs is generated.
The resulting network may be a connected or disconnected
network.

When the network is completed, i.e. the arc tree is filled
with arcs (random pairs of numbers), the network is written
to a file.

Two types of run-time errors may occur:

   101 Disk write error, if the disk becomes full.
   203 Heap overflow error. Each dynamic variable 
       is stored in the heap by stacking them on the
       top of each other.
       
3. EXAMPLE OUTPUT DATA FILES

EXAMPLE: A Fully Dense Acyclic Network
with Random Arc Capacities

c Fully Dense Acyclic Network
c for Max-Flow
c Arcs with random capacities
p max         5        10
n             1  s
n             5  t
a         1         2         7
a         1         3         3
a         1         4         6
a         1         5         5
a         2         3         5
a         2         4         8
a         2         5         2
a         3         4         7
a         3         5         4
a         4         5         2


EXAMPLE: A Fully Dense Acyclic Network
with Special Arc Capacities; (Glover et al
"Hard Network", but with rounded capacities)

c Fully Dense Acyclic Network
c for Max-Flow
c Arcs with special capacities
p max         5        10
n             1  s
n             5  t
a         1         2         3
a         1         3         1
a         1         4         1
a         1         5         1
a         2         3         1
a         2         4         1
a         2         5         1
a         3         4         1
a         3         5         1
a         4         5         3


EXAMPLE: A Sparse Acyclic Network with Special
Arc Capacities; (Dinic Worst Case Network)

c Sparse Acyclic Network
c for Max-Flow
c Arcs with special capacities
p max         5         7
n             1  s
n             5  t
a         1         2         5
a         1         5         1
a         2         3         5
a         2         5         1
a         3         4         5
a         3         5         1
a         4         5         1


EXAMPLE: A Bipartite Network with Unit Capacities

c Bipartite Network for Max-Flow
c with UNIT capacities on bipartite arcs
p max         8        15
n             1  s
n             8  t
a         1         2        11
a         1         3        25
a         1         4         7
a         2         5         1
a         2         6         1
a         2         7         1
a         3         5         1
a         3         6         1
a         3         7         1
a         4         5         1
a         4         6         1
a         4         7         1
a         5         8        10
a         6         8        15
a         7         8        30


EXAMPLE: A Bipartite Network with Random Capacities

c Bipartite Network for Max-Flow
c with RANDOM capacities
p max         8        15
n             1  s
n             8  t
a         1         2        11
a         1         3        30
a         1         4        22
a         2         5         7
a         2         6         8
a         2         7         4
a         3         5         2
a         3         6        10
a         3         7         2
a         4         5        23
a         4         6        27
a         4         7        27
a         5         8        23
a         6         8        18
a         7         8        19


EXAMPLE: One-Way Transit Grid Network
         Random Arc Capacities

c One-Way Transit Grid Network
c for Max-Flow
p max         6         8
n             1  s
n             6  t
a         1         2         8
a         1         3         5
a         2         3         1
a         2         4         8
a         3         5         9
a         5         4         6
a         4         6        10
a         5         6         5


EXAMPLE: Two-Way Transit Grid Network
         Random Arc Capacities

c Two-Way Transit Grid Network
c for Max-Flow
p max         6        16
n             1  s
n             6  t
a         1         2        10
a         2         1         3
a         1         3         7
a         3         1         1
a         2         3        10
a         3         2         9
a         2         4         7
a         4         2         6
a         3         5         5
a         5         3         3
a         4         5         9
a         5         4         2
a         4         6         5
a         6         4         6
a         5         6         6
a         6         5        10


EXAMPLE: Random Network
         Random Arc Capacities

c Random Network
c for Max-Flow
p max         5        10
n             1  s
n             5  t
a         5         3        30
a         4         2        23
a         3         5        16
a         1         3         7
a         2         4         6
a         4         3         6
a         2         1        25
a         1         4        30
a         4         1        23
a         2         5        18
SHAR_EOF
cat << 'SHAR_EOF' > waissi/tp55
acmaxi.pasbimaxi.pasramaxi.pasSHAR_EOF
cat << 'SHAR_EOF' > waissi/tr1-max.p
{
 TRANSIT GRID NETWORK GENERATOR FOR MAX-FLOW/
 ONE-WAY SYSTEM              (revised 11/25/90)
                             (revised 01/05/91)

 Copyright (C) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128

 Tel: (313) 593-5012
 E-mail: gary_waissi@um.cc.mich.edu

 Revisions: INTEGER changed to LONGINT to allow the
 range [-2^31 ... 2^31-1].

 This program generates ONE-WAY TRANSIT GRID NETWORKS
 for MAX-FLOW problems to user specified files with 
 at most one arc between any pair of nodes. The 
 direction of a generated arc is randomly assigned.
 The key procedure is:

 PROCEDURE TransitOneNet;

 Data file format corresponds to DIMACS data file 
 specifications described in "The First DIMACS 
 International Algorithm Implementation Challenge:
 Problem Definitions and Specifications". Please
 contact via e-mail: netflow@dimacs.rutgers.edu

 This program is written in standard Pascal for
 transportability.

 Another version, that uses windowing and a user 
 friendly menu interface is available. The version
 uses Turbo Pascal 5.5 features (including CTR, DOS),
 and is compiled in Turbo Pascal Units TPU's. Runs on
 DOS 3.xx and latter versions. Five network generators
 and a new max-flow algorithm are included in the 
 package, and can be invoked from the menu.

 This program is robust. A generated arc is written to
 the disk file, and not stored in the memory. Disk full
 causes a run-time error.
}
Program TransitGridNetworkOne(input,output);

  type string = packed array [1..20] of char;

    var ii,jj,kk,cap:integer;
      i,z,k,num,col,imax,zmax,colmax,capacity:integer;
      num_nodes,num_arcs:integer;
      source,sink:integer;
      ch:char;
      found:boolean;
      OutFileName:string;
      netres:text;

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
      writeln(netres,ii:10,jj:10,cap:10);
    End;
  
  Procedure GetArc(var head:integer);
    Begin
      ii:=i; jj:=head;
      cap:=irandom(capacity)+1;
    End;

  Procedure TransitOneNet;
    Procedure ChangeDir;
      var a,test:integer;
      Begin
        test:=irandom(capacity);
        if (test<0.50*capacity) then
          Begin a:=ii; ii:=jj; jj:=a;
          End;
      End;

    Procedure Gen1;
      Begin
        for z:=1 to colmax do
        Begin
          i:=1;
          kk:=z+1;
          GetArc(kk); WriteArc;
        End;
      End;

    Procedure Gen2;
      Begin
        i:=imax-colmax+2;
        for z:=(zmax+colmax+1) to (zmax+2*colmax) do
        Begin
          kk:=imax+2; GetArc(kk); WriteArc;
          i:=i+1;
        End;
      End;

    Procedure GenArcs;
      Procedure DoArc;
        Begin
          GetArc(kk); ChangeDir; WriteArc;
        End;

      Begin
        col:=1; num:=colmax; i:=2; z:=colmax+1;
        repeat
          repeat
            if (i<num+1) then
              Begin kk:=i+1; DoArc; z:=z+1;
              End;
            kk:=i+colmax;
            DoArc;
            z:=z+1; i:=i+1;
          until (i=num+2);
          num:=num+colmax;
          col:=col+1;
        until (col=colmax);
        repeat kk:=i+1; DoArc; i:=i+1; z:=z+1;
        until (i=imax+1);
      End;

    Begin         { Procedure TransitOneNet; }
      Gen1; GenArcs; Gen2
    End;
 
  Procedure Banner1;
    Begin
      writeln;
      writeln('    ONE-WAY TRANSIT GRID NETWORK');
      writeln('            GENERATOR');
      writeln('    Copyright (C) Gary R. Waissi');
      writeln('              (1990)');
      writeln('   University of Michigan-Dearborn');
      writeln('     School of Management, 113FOB');
      writeln('        Dearborn, MI 48128');
      writeln;
      writeln('  This program generates a ONE-WAY');
      writeln('  TRANSIT GRID network for MAX-FLOW');
      writeln('  into a user file. The network');
      writeln('  resembles a one-way city street');
      writeln('  network. The direction of each ');
      writeln('  one-way street, except those');
      writeln('  connected to the common source');
      writeln('  and common sink respectively, is');
      writeln('  randomly assigned.');
      writeln;
    End;

  Procedure Banner2;
    Begin
      writeln;
      writeln('  For the number of nodes in the');
      writeln('  network select a number that ');
      writeln('  creates a complete square grid.');
      writeln('  For example:');
      writeln('      4 nodes in a 2 x 2 network,');
      writeln('      9 nodes in a 3 x 3 network,');
      writeln('     16 nodes in a 4 x 4 network,');
      writeln('     25 nodes in a 5 x 5 network,');
      writeln('   ...');
      writeln('   1600 nodes in a 40 x 40 network.');
      writeln('  The program adds automatically');
      writeln('  two nodes, a supersource and ');
      writeln('  a supersink, to the network.');
    End;

  Procedure Banner3;
    Begin
      writeln;
      writeln('  A ONE-WAY TRANSIT GRID NETWORK');
      writeln('  for MAX-FLOW will be generated');
      writeln('  to file ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The source node is node ',sink:8);
      writeln('  The sink node is numbered with the');
      writeln('  largest node number.');
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
      writeln;
    End;

  Procedure Banner4;
    Begin
      writeln;
      writeln('  The network is completed. The');
      writeln('  file is an ASCII file, with name:');
	  writeln('  ',OutFileName,',and can be');
      writeln('  accessed with your editor.');
    End;

  Procedure UserValues;
    Begin
      repeat
        writeln('  How many nodes are in the network ?');
        write('  (4, 9, 16, 25,..., 400, 625,...): ');
        readln(imax);
        if (imax<4) then
        Begin
          writeln('  Try again. Need at least four');
          writeln('  nodes in the smallest grid ');
          writeln('  network.');
        End;
      until (imax>=4);
      colmax:=round(sqrt(imax));
      zmax:=2*(imax-colmax);
      writeln('  What is the upper bound for arc flow');
      write('  CAPACITY: ');
      readln(capacity);
      num_nodes:=imax+2;
      num_arcs:=zmax+2*colmax;
      source:=1;
      sink:=num_nodes;
      Banner3;
      writeln(netres,'c One-Way Transit Grid Network');
      writeln(netres,'c for Max-Flow');
      writeln(netres,'p max',num_nodes:10,num_arcs:10);
      writeln(netres,'n',source:14,'  s');
      writeln(netres,'n',sink:14,'  t');
    End;

  Begin
    randomize; {Turbo Pascal random number generator initiator.}
               {You may to replace this by compiler specific   }
               {randomizer, or write a short procedure.        }
    Banner1;
    writeln('  Try the network generation out by');
    writeln('  following the instructions.');
    write('  Do you want to continue (Y/N) ');
    readln(ch);
    if (ch='y') or (ch='Y') then
    Begin
      write('  Enter name of the output file: ');
      readln(OutFileName);
{ SUN       assign(netres,OutFileName+'.max'); }
      rewrite(netres,OutFileName);
      Banner2;
      UserValues;
      TransitOneNet;
{ SUN       close(netres); }
      Banner4;
    End;
  End.
SHAR_EOF
cat << 'SHAR_EOF' > waissi/tr1-max.pas
{
 TRANSIT GRID NETWORK GENERATOR FOR MAX-FLOW/
 ONE-WAY SYSTEM              (revised 11/25/90)
                             (revised 01/05/91)

 Copyright (C) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128

 Tel: (313) 593-5012
 E-mail: gary_waissi@um.cc.mich.edu

 Revisions: INTEGER changed to LONGINT to allow the
 range [-2^31 ... 2^31-1].

 This program generates ONE-WAY TRANSIT GRID NETWORKS
 for MAX-FLOW problems to user specified files with 
 at most one arc between any pair of nodes. The 
 direction of a generated arc is randomly assigned.
 The key procedure is:

 PROCEDURE TransitOneNet;

 Data file format corresponds to DIMACS data file 
 specifications described in "The First DIMACS 
 International Algorithm Implementation Challenge:
 Problem Definitions and Specifications". Please
 contact via e-mail: netflow@dimacs.rutgers.edu

 This program is written in standard Pascal for
 transportability.

 Another version, that uses windowing and a user 
 friendly menu interface is available. The version
 uses Turbo Pascal 5.5 features (including CTR, DOS),
 and is compiled in Turbo Pascal Units TPU's. Runs on
 DOS 3.xx and latter versions. Five network generators
 and a new max-flow algorithm are included in the 
 package, and can be invoked from the menu.

 This program is robust. A generated arc is written to
 the disk file, and not stored in the memory. Disk full
 causes a run-time error.
}
Program TransitGridNetworkOne(input,output);

  var ii,jj,kk,cap:longint;
      i,z,k,num,col,imax,zmax,colmax,capacity:longint;
      num_nodes,num_arcs:longint;
      source,sink:longint;
      ch:char;
      found:boolean;
      OutFileName:string[20];
      netres:text;
  
  Procedure WriteArc;
    Begin
      write(netres,'a');
      writeln(netres,ii:10,jj:10,cap:10);
    End;
  
  Procedure GetArc(var head:longint);
    Begin
      ii:=i; jj:=head;
      cap:=random(capacity)+1;
    End;

  Procedure TransitOneNet;
    Procedure ChangeDir;
      var a,test:longint;
      Begin
        test:=random(capacity);
        if (test<0.50*capacity) then
          Begin a:=ii; ii:=jj; jj:=a;
          End;
      End;

    Procedure Gen1;
      Begin
        for z:=1 to colmax do
        Begin
          i:=1;
          kk:=z+1;
          GetArc(kk); WriteArc;
        End;
      End;

    Procedure Gen2;
      Begin
        i:=imax-colmax+2;
        for z:=(zmax+colmax+1) to (zmax+2*colmax) do
        Begin
          kk:=imax+2; GetArc(kk); WriteArc;
          i:=i+1;
        End;
      End;

    Procedure GenArcs;
      Procedure DoArc;
        Begin
          GetArc(kk); ChangeDir; WriteArc;
        End;

      Begin
        col:=1; num:=colmax; i:=2; z:=colmax+1;
        repeat
          repeat
            if (i<num+1) then
              Begin kk:=i+1; DoArc; z:=z+1;
              End;
            kk:=i+colmax;
            DoArc;
            z:=z+1; i:=i+1;
          until (i=num+2);
          num:=num+colmax;
          col:=col+1;
        until (col=colmax);
        repeat kk:=i+1; DoArc; i:=i+1; z:=z+1;
        until (i=imax+1);
      End;

    Begin         { Procedure TransitOneNet; }
      Gen1; GenArcs; Gen2
    End;
 
  Procedure Banner1;
    Begin
      writeln;
      writeln('    ONE-WAY TRANSIT GRID NETWORK');
      writeln('            GENERATOR');
      writeln('    Copyright (C) Gary R. Waissi');
      writeln('              (1990)');
      writeln('   University of Michigan-Dearborn');
      writeln('     School of Management, 113FOB');
      writeln('        Dearborn, MI 48128');
      writeln;
      writeln('  This program generates a ONE-WAY');
      writeln('  TRANSIT GRID network for MAX-FLOW');
      writeln('  into a user file. The network');
      writeln('  resembles a one-way city street');
      writeln('  network. The direction of each ');
      writeln('  one-way street, except those');
      writeln('  connected to the common source');
      writeln('  and common sink respectively, is');
      writeln('  randomly assigned.');
      writeln;
    End;

  Procedure Banner2;
    Begin
      writeln;
      writeln('  For the number of nodes in the');
      writeln('  network select a number that ');
      writeln('  creates a complete square grid.');
      writeln('  For example:');
      writeln('      4 nodes in a 2 x 2 network,');
      writeln('      9 nodes in a 3 x 3 network,');
      writeln('     16 nodes in a 4 x 4 network,');
      writeln('     25 nodes in a 5 x 5 network,');
      writeln('   ...');
      writeln('   1600 nodes in a 40 x 40 network.');
      writeln('  The program adds automatically');
      writeln('  two nodes, a supersource and ');
      writeln('  a supersink, to the network.');
    End;

  Procedure Banner3;
    Begin
      writeln;
      writeln('  A ONE-WAY TRANSIT GRID NETWORK');
      writeln('  for MAX-FLOW will be generated');
      writeln('  to file ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The source node is node ',sink:8);
      writeln('  The sink node is numbered with the');
      writeln('  largest node number.');
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
      writeln;
    End;

  Procedure Banner4;
    Begin
      writeln;
      writeln('  The network is completed. The');
      writeln('  data file is an ASCII file, and');
      writeln('  can be accessed with your editor.');
      writeln('  You may print the file using the');
      writeln('  DOS command PRINT ',OutFileName,'.max.');
    End;

  Procedure UserValues;
    Begin
      repeat
        writeln('  How many nodes are in the network ?');
        write('  (4, 9, 16, 25,..., 400, 625,...): ');
        readln(imax);
        if (imax<4) then
        Begin
          writeln('  Try again. Need at least four');
          writeln('  nodes in the smallest grid ');
          writeln('  network.');
        End;
      until (imax>=4);
      colmax:=round(sqrt(imax));
      zmax:=2*(imax-colmax);
      writeln('  What is the upper bound for arc flow');
      write('  CAPACITY: ');
      readln(capacity);
      num_nodes:=imax+2;
      num_arcs:=zmax+2*colmax;
      source:=1;
      sink:=num_nodes;
      Banner3;
      writeln(netres,'c One-Way Transit Grid Network');
      writeln(netres,'c for Max-Flow');
      writeln(netres,'p max',num_nodes:10,num_arcs:10);
      writeln(netres,'n',source:14,'  s');
      writeln(netres,'n',sink:14,'  t');
    End;

  Begin
    randomize; {Turbo Pascal random number generator initiator.}
               {You may to replace this by compiler specific   }
               {randomizer, or write a short procedure.        }
    Banner1;
    writeln('  Try the network generation out by');
    writeln('  following the instructions.');
    write('  Do you want to continue (Y/N) ');
    readln(ch);
    if (ch='y') or (ch='Y') then
    Begin
      write('  Enter name of the output file: ');
      readln(OutFileName);
      assign(netres,OutFileName+'.max');
      rewrite(netres);
      Banner2;
      UserValues;
      TransitOneNet;
      close(netres);
      Banner4;
    End;
  End.
SHAR_EOF
cat << 'SHAR_EOF' > waissi/tr2-max.p
{
 TRANSIT GRID NETWORK GENERATOR FOR MAX-FLOW/
 TWO-WAY SYSTEM             (revised 11/25/90)
                            (revised 01/05/91)

 Copyright (C) (1990) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128

 Tel: (313) 593-5012
 E-mail: gary_waissi@um.cc.mich.edu

 Revision: INTEGER changed to LONGINT to allow input
 value range of [-2^31 ... 2^31-1].

 This program generates TWO-WAY TRANSIT GRID NETWORKS
 for MAX-FLOW to user specified files with exactly
 two arcs between any pair of nodes that form the grid.
 Key procedure is:

 PROCEDURE TransitTwoNet;

 Data file format corresponds to DIMACS data file 
 specifications described in "The First DIMACS 
 International Algorithm Implementation Challenge:
 Problem Definitions and Specifications". Please 
 contact via e-mail: netflow@dimacs.rutgers.edu

 This program is written in standard Pascal for
 transportability.

 Another version, that uses windowing and a user
 friendly menu interface is available. The version
 uses Turbo Pascal 5.5 features (including CTR,
 DOS), and is compiled in Turbo Pascal Units TPU's.
 Runs on DOS 3.xx and latter versions. Five network
 generators and a new max-flow algorithm are 
 included in the package, and can be invoked from 
 the menu.

 This program is robust. A generated arc is written
 to the disk file, and not stored in the memory.
 Disk full causes a run-time error.
}

Program TransitGridNetworkTwo(input,output);

   type string = packed array [1..20] of char;
  var ii,jj,kk,zz,cap:integer;
      i,z,k,num,col,imax,zmax,colmax,capacity:integer;
      num_nodes,num_arcs:integer;
      source,sink:integer;
      ch:char;
      found:boolean;
      OutFileName:string;
      netres:text;
 
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
      writeln(netres,ii:10,jj:10,cap:10);
      zz:=zz+1;
    End;

  Procedure GetArc1(var head:integer);
    Begin
      ii:=i; jj:=head;
      cap:=irandom(capacity)+1;
    End;

  Procedure GetArc2(var head:integer);
    Begin
      ii:=head; jj:=i;
      cap:=irandom(capacity)+1;
    End;

  Procedure TransitTwoNet;
    Procedure Gen1;
      Begin
        for z:=1 to colmax do
        Begin
          i:=1; kk:=z+1;
          GetArc1(kk);
          WriteArc;
          GetArc2(kk);
          WriteArc;
        End;
      End;

    Procedure Gen2;
      Begin
        i:=imax-colmax+2;
        for z:=(zmax+colmax+1) to (zmax+2*colmax) do
        Begin
          kk:=imax+2;
          GetArc1(kk); WriteArc;
          GetArc2(kk); WriteArc;
          i:=i+1;
        End;
      End;

    Procedure GenArcs;
      Procedure DoArc;
        Begin
          GetArc1(kk); WriteArc;
          GetArc2(kk); WriteArc;
        End;

      Begin
        col:=1; num:=colmax; i:=2; z:=colmax+1;
        repeat
          repeat
            if (i<num+1) then
              Begin kk:=i+1; DoArc; z:=z+1;
              End;
            kk:=i+colmax;
            DoArc;
            z:=z+1; i:=i+1;
          until (i=num+2);
          num:=num+colmax;
          col:=col+1;
        until (col=colmax);
        repeat kk:=i+1; DoArc; i:=i+1; z:=z+1;
        until (i=imax+1);
      End;

    Begin         { Procedure TransitTwoNet; }
      zz:=1; Gen1; GenArcs; Gen2
    End;
  
  Procedure Banner1;
    Begin
      writeln;
      writeln('    TWO-WAY TRANSIT GRID NETWORK');
      writeln('       GENERATOR FOR MAX-FLOW');
      writeln('    Copyright (C) Gary R. Waissi');
      writeln('              (1990)');
      writeln('   University of Michigan-Dearborn');
      writeln('     School of Management, 113FOB');
      writeln('        Dearborn, MI 48128');
      writeln;
      writeln('  This program generates a TWO-WAY');
      writeln('  TRANSIT GRID network for MAX-FLOW');
      writeln('  into a user file. The network ');
      writeln('  resembles a two-way city street');
      writeln('  network.');
      writeln;
    End;

  Procedure Banner2;
    Begin
      writeln;
      writeln('  For the number of nodes select ');
      writeln('  a number that creates a complete');
      writeln('  square grid. For example:');
      writeln('       4 nodes in a 2 x 2 network,');
      writeln('       9 nodes in a 3 x 3 network,');
      writeln('      16 nodes in a 4 x 4 network,');
      writeln('      25 nodes in a 5 x 5 network,');
      writeln('    ...');
      writeln('    1600 nodes in a 40 x 40 network.');
      writeln('  The program adds automatically ');
      writeln('  two nodes, a supersource and ');
      writeln('  a supersink, to the network.');
      writeln;
    End;

  Procedure Banner3;
    Begin
      writeln;
      writeln('  A TWO-WAY TRANSIT GRID NETWORK for');
      writeln('  MAX-FLOW will be generated to file');
      writeln('    ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The sink node is node   ',sink:8);
      writeln('  The sink node is numbered with the');
      writeln('  largest node number.');
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
      writeln;
    End;

  Procedure Banner4;
    Begin
      writeln;
      writeln('  The network is completed. The data');
      writeln('  file is an ASCII file, with name:');
	  writeln('  ',OutFileName,',and can be');
      writeln('  accessed with your editor.');
    End;

  Procedure UserValues;
    Begin
      repeat
        writeln('  How many nodes are in the network: ');
        write('  (4, 9, 16, 25, ..., 400, 625,...):');
        readln(imax);
        if (imax<4) then
        Begin
          writeln('  Try again. Need at least four');
          writeln('  nodes in the smallest grid network.');
        End;
      until (imax>=4);
      colmax:=round(sqrt(imax));
      zmax:=2*(imax-colmax);
      writeln('  What is the upper bound for arc flow');
      write('  CAPACITY: ');
      readln(capacity);
      num_nodes:=imax+2;
      num_arcs:=2*(zmax+2*colmax);
      source:=1;
      sink:=num_nodes;
      Banner3;
      writeln(netres,'c Two-Way Transit Grid Network');
      writeln(netres,'c for Max-Flow');
      writeln(netres,'p max',num_nodes:10,num_arcs:10);
      writeln(netres,'n',source:14,'  s');
      writeln(netres,'n',sink:14,'  t');
    End;

  Begin
    randomize;  {Turbo Pascal random number generator initiator.}
                {You may to replace this by compiler specific   }
                {randomizer, or write a short procedure.        }
    Banner1;
    writeln('  Try the network generation out by');
    writeln('  following the instructions.');
    write('  Do you want to continue (Y/N) ');
    readln(ch);
    if (ch='y') or (ch='Y') then
    Begin
      write('  Enter name of the output file: ');
      readln(OutFileName);
{ SUN       assign(netres,OutFileName+'.max'); }
      rewrite(netres,OutFileName);
      Banner2;
      UserValues;
      TransitTwoNet;
{ SUN       close(netres); }
      Banner4;
    End;
  End.
SHAR_EOF
cat << 'SHAR_EOF' > waissi/tr2-max.pas
{
 TRANSIT GRID NETWORK GENERATOR FOR MAX-FLOW/
 TWO-WAY SYSTEM             (revised 11/25/90)
                            (revised 01/05/91)

 Copyright (C) (1990) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128

 Tel: (313) 593-5012
 E-mail: gary_waissi@um.cc.mich.edu

 Revision: INTEGER changed to LONGINT to allow input
 value range of [-2^31 ... 2^31-1].

 This program generates TWO-WAY TRANSIT GRID NETWORKS
 for MAX-FLOW to user specified files with exactly
 two arcs between any pair of nodes that form the grid.
 Key procedure is:

 PROCEDURE TransitTwoNet;

 Data file format corresponds to DIMACS data file 
 specifications described in "The First DIMACS 
 International Algorithm Implementation Challenge:
 Problem Definitions and Specifications". Please 
 contact via e-mail: netflow@dimacs.rutgers.edu

 This program is written in standard Pascal for
 transportability.

 Another version, that uses windowing and a user
 friendly menu interface is available. The version
 uses Turbo Pascal 5.5 features (including CTR,
 DOS), and is compiled in Turbo Pascal Units TPU's.
 Runs on DOS 3.xx and latter versions. Five network
 generators and a new max-flow algorithm are 
 included in the package, and can be invoked from 
 the menu.

 This program is robust. A generated arc is written
 to the disk file, and not stored in the memory.
 Disk full causes a run-time error.
}

Program TransitGridNetworkTwo(input,output);

  var ii,jj,kk,zz,cap:longint;
      i,z,k,num,col,imax,zmax,colmax,capacity:longint;
      num_nodes,num_arcs:longint;
      source,sink:longint;
      ch:char;
      found:boolean;
      OutFileName:string[20];
      netres:text;
  
  Procedure WriteArc;
    Begin
      write(netres,'a');
      writeln(netres,ii:10,jj:10,cap:10);
      zz:=zz+1;
    End;

  Procedure GetArc1(var head:longint);
    Begin
      ii:=i; jj:=head;
      cap:=random(capacity)+1;
    End;

  Procedure GetArc2(var head:longint);
    Begin
      ii:=head; jj:=i;
      cap:=random(capacity)+1;
    End;

  Procedure TransitTwoNet;
    Procedure Gen1;
      Begin
        for z:=1 to colmax do
        Begin
          i:=1; kk:=z+1;
          GetArc1(kk);
          WriteArc;
          GetArc2(kk);
          WriteArc;
        End;
      End;

    Procedure Gen2;
      Begin
        i:=imax-colmax+2;
        for z:=(zmax+colmax+1) to (zmax+2*colmax) do
        Begin
          kk:=imax+2;
          GetArc1(kk); WriteArc;
          GetArc2(kk); WriteArc;
          i:=i+1;
        End;
      End;

    Procedure GenArcs;
      Procedure DoArc;
        Begin
          GetArc1(kk); WriteArc;
          GetArc2(kk); WriteArc;
        End;

      Begin
        col:=1; num:=colmax; i:=2; z:=colmax+1;
        repeat
          repeat
            if (i<num+1) then
              Begin kk:=i+1; DoArc; z:=z+1;
              End;
            kk:=i+colmax;
            DoArc;
            z:=z+1; i:=i+1;
          until (i=num+2);
          num:=num+colmax;
          col:=col+1;
        until (col=colmax);
        repeat kk:=i+1; DoArc; i:=i+1; z:=z+1;
        until (i=imax+1);
      End;

    Begin         { Procedure TransitTwoNet; }
      zz:=1; Gen1; GenArcs; Gen2
    End;
  
  Procedure Banner1;
    Begin
      writeln;
      writeln('    TWO-WAY TRANSIT GRID NETWORK');
      writeln('       GENERATOR FOR MAX-FLOW');
      writeln('    Copyright (C) Gary R. Waissi');
      writeln('              (1990)');
      writeln('   University of Michigan-Dearborn');
      writeln('     School of Management, 113FOB');
      writeln('        Dearborn, MI 48128');
      writeln;
      writeln('  This program generates a TWO-WAY');
      writeln('  TRANSIT GRID network for MAX-FLOW');
      writeln('  into a user file. The network ');
      writeln('  resembles a two-way city street');
      writeln('  network.');
      writeln;
    End;

  Procedure Banner2;
    Begin
      writeln;
      writeln('  For the number of nodes select ');
      writeln('  a number that creates a complete');
      writeln('  square grid. For example:');
      writeln('       4 nodes in a 2 x 2 network,');
      writeln('       9 nodes in a 3 x 3 network,');
      writeln('      16 nodes in a 4 x 4 network,');
      writeln('      25 nodes in a 5 x 5 network,');
      writeln('    ...');
      writeln('    1600 nodes in a 40 x 40 network.');
      writeln('  The program adds automatically ');
      writeln('  two nodes, a supersource and ');
      writeln('  a supersink, to the network.');
      writeln;
    End;

  Procedure Banner3;
    Begin
      writeln;
      writeln('  A TWO-WAY TRANSIT GRID NETWORK for');
      writeln('  MAX-FLOW will be generated to file');
      writeln('    ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The sink node is node   ',sink:8);
      writeln('  The sink node is numbered with the');
      writeln('  largest node number.');
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
      writeln;
    End;

  Procedure Banner4;
    Begin
      writeln;
      writeln('  The network is completed. The data');
      writeln('  file is an ASCII file, and can be');
      writeln('  accessed with your editor. You may');
      writeln('  print the file using the DOS');
      writeln('  command PRINT ',OutFileName,'.max.');
    End;

  Procedure UserValues;
    Begin
      repeat
        writeln('  How many nodes are in the network: ');
        write('  (4, 9, 16, 25, ..., 400, 625,...):');
        readln(imax);
        if (imax<4) then
        Begin
          writeln('  Try again. Need at least four');
          writeln('  nodes in the smallest grid network.');
        End;
      until (imax>=4);
      colmax:=round(sqrt(imax));
      zmax:=2*(imax-colmax);
      writeln('  What is the upper bound for arc flow');
      write('  CAPACITY: ');
      readln(capacity);
      num_nodes:=imax+2;
      num_arcs:=2*(zmax+2*colmax);
      source:=1;
      sink:=num_nodes;
      Banner3;
      writeln(netres,'c Two-Way Transit Grid Network');
      writeln(netres,'c for Max-Flow');
      writeln(netres,'p max',num_nodes:10,num_arcs:10);
      writeln(netres,'n',source:14,'  s');
      writeln(netres,'n',sink:14,'  t');
    End;

  Begin
    randomize;  {Turbo Pascal random number generator initiator.}
                {You may to replace this by compiler specific   }
                {randomizer, or write a short procedure.        }
    Banner1;
    writeln('  Try the network generation out by');
    writeln('  following the instructions.');
    write('  Do you want to continue (Y/N) ');
    readln(ch);
    if (ch='y') or (ch='Y') then
    Begin
      write('  Enter name of the output file: ');
      readln(OutFileName);
      assign(netres,OutFileName+'.max');
      rewrite(netres);
      Banner2;
      UserValues;
      TransitTwoNet;
      close(netres);
      Banner4;
    End;
  End.
SHAR_EOF
:	End of shell archive
exit 0
