{
 ACYCLIC NETWORK GENERATOR FOR MAX-FLOW 

 Copyright (C) (1990,1991) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128
 See file README1.TXT for details.
}
{$O+,F+}
UNIT AcMaxi;

INTERFACE
  var ch:char;
      OutFileName:string[20];
      DoContinue:boolean;

  Procedure AcMaxiNet;

IMPLEMENTATION

  uses CRT;

  var
    cap:real;
    y,p,q,i,k,num,try,arcnumber:longint;
    node_i,node_j,num_arcs,num_nodes,capacity:longint;
    tail,head:longint;
    done,special,sparse:boolean;
    netres:text;
    source:longint;
    sink:longint;
    delay1:integer;
    IOCode:integer;

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

Procedure ReadOutFileName;
  Begin 
    write('  Enter name of the output file: ');
    repeat
      {$I-}
      readln(OutFileName);
      assign(netres,OutFileName+'.max');
      rewrite(netres);
      IOCode:=IOResult;
      if (IOCode<>0) then
      begin
        writeln('  Output file name must be a legal');
        writeln('  DOS filename. The program appends');
        writeln('  the filename automatically with');
        writeln('  an extension .max. Please do not');
        write('  use any file extensions. FILE: ');
      end;
    until (IOCode=0);
  End;

Procedure ReadResponse;
  Begin
    repeat
      repeat
        {$I-}
        readln(ch);
        {$I+}
        IOCode:=IOResult;
        if IOCode<>0 then write('  Try again ! (Y/N) ')
      until IOCode=0;
      if not ((ch='y') or (ch='Y') or (ch='n') or (ch='N'))
        then write('  Try again ! (Y or N) ');
    until (ch='y') or (ch='Y') or (ch='n') or (ch='N');
  End;

Procedure ModifyDelay;
  Begin
    delay1:=15000;
    writeln('  Introductory information about the');
    writeln('  program is displayed on a few pages');
    writeln('  for 15 seconds per page. Do you');
    writeln('  want to change the time/page the');
    write('  information is displayed? (Y/N): ');
    ReadResponse;
    if (ch='y') or (ch='Y') then
    Begin
      write('  Enter a new time/page in seconds: ');
      repeat
        {$I-}
        readln(delay1);
        {$I+}
        IOCode:=IOResult;
        if IOCode<>0 then write('  Type an integer ! (seconds): ');
      until IOCode=0;
      delay1:=delay1*1000;
    End;
  End;

  Procedure Banner1;
  Begin
    clrscr;
    writeln('     ACYCLIC MAX-FLOW NETWORK');
    writeln('           GENERATOR');
    writeln;
    writeln('  The program generates an ACYCLIC');
    writeln('  MAX-FLOW network into a user file.');
    writeln('  An acyclic network is such that');
    writeln('  for each arc (i,j), where i is the');
    writeln('  arc tail node number, and j is the');
    writeln('  arc head node number i<j.');
    writeln('  The generated acyclic network will');
    writeln('  be either fully dense or sparse.');
    writeln;
    writeln('  Try the network generation out by');
    writeln('  following the instructions.');
    writeln;
    write('  Do you want to continue (Y/N) ');
    ReadResponse;
    clrscr;
  End;
  
  Procedure Banner2;
    Begin
      clrscr;
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
      write('  Please WAIT !!');
      Delay(delay1);
      clrscr;
      writeln('  forces flow on each arc to capacity');
      writeln('  at optimum. The networks were');
      writeln('  called in the study "hard networks",');
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
      write('  Please WAIT !!');
      Delay(delay1);
      clrscr;
      writeln('  E.A., Algorithm for Solution of ');
      writeln('  a Problem of Maximum Flow in ');
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
      writeln;
    End;
  
  Procedure Banner3;
    Begin
      clrscr;
      writeln('  You can choose either random');
      writeln('  capacities for arcs, or special');
      writeln('  arc capacities. Special arc ');
      writeln('  capacities are calculated using');
      writeln('  the following (Glover et al):');
      writeln;
      writeln('  1. cap(i,j)=1 for each (i,j) with');
      writeln('     j>i+1');
      writeln('  2. cap(i,j)=1+(i-n/2)^2 for each');
      writeln('     (i,j) with j=i+1');
      writeln;
      writeln('  The program, however, will round');
      writeln('  all non-integer capacities to');
      writeln('  integer values to conform with');
      writeln('  DIMACS specifications.');
      write('  Please WAIT !!');
      Delay(delay1);
      clrscr;
    End;

  Procedure Banner4;
    Begin
      clrscr;
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
      writeln('  Please notice, that for simplicity');
      writeln('  the source node s is always numbered');
      writeln('  with 1, and the sink node t is');
      writeln('  assigned the largest node number,');
      writeln('  here t = ',sink,'.');
      writeln;
      writeln('  AN ACYCLIC MAX-FLOW NETWORK will');
      writeln('  be generated to file ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The sink node is node   ',sink:8);
      writeln('  The network is given as a list of');
      writeln('  lines: comment lines, problem line,');
      writeln('  node lines, arc lines as follows:');
      write('  Please WAIT !!');
      Delay(delay1);
      clrscr;
      writeln('  c This is a comment line.');
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
      writeln;
      write('  Please WAIT !!');
      Delay(delay1);
      clrscr;
      write('  Please WAIT !!');
    End;

  Procedure Banner5;
    Begin
      writeln;
      writeln('  The network is completed. The data');
      writeln('  file is an ASCII file, and can be');
      writeln('  accessed with your editor. You may');
      writeln('  print the file by first returning');;
      writeln('  to DOS and then using the DOS');
      writeln('  command PRINT ',OutFileName,'.max.');
      writeln;
      write('  PRESS ENTER TO RETURN TO MAIN MENU. ');
      readln;
    End;

  Procedure UserValues;
    Begin
      sparse:=true;
      special:=false;
      done:=false;
      try:=1;
      writeln;
      write('  How many nodes are in the network: ');
      repeat
        repeat
          {$I-}
          readln(num_nodes);
          {$I+}
          IOCode:=IOResult;
          if IOCode<>0 then write('  Try again ! Number of nodes: ');
        until IOCode=0;
        if (num_nodes<=1) then
        write('  Try again. Need at least two nodes: ');
      until (num_nodes>1);
      source:=1;
      sink:=num_nodes;
      Banner2;
      writeln('  Do you want a FULLY DENSE or');
      writeln('  a SPARSE acyclic network.');
      writeln('  Type F for FULLY DENSE and S for');
      write('  SPARSE (F/S): ');
      repeat
        repeat
          {$I-}
          readln(ch);
          {$I+}
          IOCode:=IOResult;
          if IOCode<>0 then write('  Try again ! (F/S): ');
        until IOCode=0;
        if not ((ch='f') or (ch='F') or (ch='s') or (ch='S'))
          then write('  Try again ! (F or S): ');
      until (ch='f') or (ch='F') or (ch='s') or (ch='S');
      if (ch='f') or (ch='F') then
      Begin
        sparse:=false;
        num_arcs:=0;
        for i:=1 to (num_nodes-1) do num_arcs:=num_arcs+i;
        Banner3;
        writeln('  Do you want special arc flow');
        write('  capacities (Y/N): ');
        ReadResponse;
        if (ch='y') or (ch='Y') then special:=true
        else
          begin
            writeln('  What is the upper bound for arc');
            write('  FLOW CAPACITY: ');
            repeat
              {$I-}
              readln(capacity);
              {$I+}
              IOCode:=IOResult;
              if IOCode<>0 then write('  Try again ! CAPACITY: ');
            until IOCode=0;
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

Procedure AcMaxiNet;
Begin
  DoContinue:=true;
  randomize;  {Turbo Pascal random number generator initiator.  }
              {You may have to replace this by compiler         }
              {specific randomizer, or write a short procedure. }
  Banner1;
  if (ch='y') or (ch='Y') then
  Begin
    ModifyDelay;
    ReadOutFileName;
    UserValues;
    close(netres);
    Banner5;
  End
  else DoContinue:=false;
End;

End.
