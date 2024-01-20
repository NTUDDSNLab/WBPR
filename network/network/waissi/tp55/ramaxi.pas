{
 RANDOM NETWORK GENERATOR FOR MAX-FLOW

 Copyright (C) (1990,1991) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128
 See file README1.TXT for details.
}
{F+}
UNIT RaMaxi;

INTERFACE

  var ch:char;
      OutFileName:string[20];
      DoContinue:boolean;

  Procedure RaMaxiNet;

IMPLEMENTATION

  uses CRT;

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
    found11,found22:boolean;
    netres:text;
    delay1:integer;
    IOCode:integer;


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
      writeln('     RANDOM NETWORK GENERATOR');
      writeln('          FOR MAX-FLOW');
      writeln;
      writeln('  This program generates a RANDOM');
      writeln('  network for MAX-FLOW into a user');
      writeln('  file. The network is generated by');
      writeln('  randomly pairing a given number');
      writeln('  of nodes using a given number of');
      writeln('  arcs.');
      writeln;
      writeln('  Try the network generation out by');
      writeln('  following the instructions.');
      write('  Do you want to continue (Y/N) ');
      ReadResponse;
      clrscr;
    End;


  Procedure Banner3;
    Begin
      clrscr;
      writeln('  A RANDOM NETWORK for MAX-FLOW will');
      writeln('  be generated to file ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The sink node is node   ',sink:8);
      writeln('  The sink node is numbered with the');
      writeln('  largest node number. ');
      writeln;
      writeln('  The network is given as a list of');
      writeln('  lines: comment lines, problem line,');
      writeln('  node lines, arc lines as follows:');
      writeln;
      writeln;
      writeln;
      write('  Please WAIT!');
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
      write('  Please WAIT!');
      Delay(delay1);
      clrscr;
      writeln('  The network generation may take');
      writeln('  some time. The entire network is');
      writeln('  temporarily stored in a tree in RAM.');
      writeln('  The storage of the tree requires');
      writeln('  storing of dynamic variables in ');
      writeln('  a heap. The heap sizes are limited,');
      writeln('  but can usually be controlled using');
      writeln('  various compiler options. ');
      writeln; writeln; writeln; writeln; writeln;
      writeln; writeln;
      write('  Please WAIT!');
      Delay(delay1);
      clrscr;
      writeln('  PLEASE WAIT !!');
      writeln;
    End;

  Procedure Banner4;
    Begin
      writeln;
      writeln('  The network is completed. The data');
      writeln('  file is an ASCII file, and can be');
      writeln('  accessed with your editor. You may');
      writeln('  print the file by first returning');
      writeln('  to DOS, and then using the DOS');
      writeln('  command PRINT ',OutFileName,'.max.');
      writeln;
      write('  PRESS ENTER TO RETURN TO MAIN MENU. ');
      readln;
    End;

  Procedure UserValues;
    var total_arcs:longint;
    Procedure NodesRead;
    Begin
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
    End;

    Procedure ArcsRead;
      Procedure Banner2;
      Begin
        clrscr;
        writeln('  The maximum number of arcs in');
        writeln('  a fully dense network cannot ');
        writeln('  exceed n^2-n. Also, you need at');
        writeln('  least one arc. Duplicate arcs');
        writeln('  and self-loops will not ');
        writeln('  be allowed.    ');
      End;

    Begin  {ArcsRead}
      repeat
        repeat
          {$I-}
          readln(num_arcs);
          {$I+}
          IOCode:=IOResult;
          if IOCode<>0 then write('  Try again ! Number of arcs: ');
        until IOCode=0;
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

    Procedure CapacityRead;
    Begin
      repeat
        {$I-}
        readln(capacity);
        {$I+}
        IOCode:=IOResult;
        if IOCode<>0 then write('  Try again ! CAPACITY: ');
      until IOCode=0;
    End;

    Begin   {UserValues}
      clrscr;
      xx1:=0;
      write('  How many nodes are in the network: ');
      NodesRead;
      total_arcs:=num_nodes*num_nodes-num_nodes;
      write('  How many arcs are in the network: ');
      ArcsRead;
      writeln('  What is the upper bound for arc flow');
      write('  CAPACITY: ');
      CapacityRead;
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

  Procedure RaMaxiNet;
  Begin
    DoContinue:=true;
    randomize;  {Turbo Pascal random number generator initiator.}
                {You may to replace this by compiler specific   }
                {randomizer, or write a short procedure.        }
    Banner1;
    if (ch='y') or (ch='Y') then
    Begin
      ModifyDelay;
      ReadOutFileName;
      UserValues;
      RandomNet;
      z:=1;
      PrintArcTree(ArcTreeRoot);
      close(netres);
      Banner4;
    End
    else DoContinue:=false;
  End;

End.
