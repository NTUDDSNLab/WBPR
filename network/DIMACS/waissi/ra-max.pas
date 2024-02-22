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
