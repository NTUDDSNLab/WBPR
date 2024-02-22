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
