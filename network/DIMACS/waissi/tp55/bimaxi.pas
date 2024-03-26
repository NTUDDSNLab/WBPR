{
 BIPARTITE MAX-FLOW NETWORK GENERATOR FOR MAX-FLOW

 Copyright (C) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128
 See file README1.TXT for details.
}
{$O+,F+}
UNIT BiMaxi;

INTERFACE

  var ch:char;
  OutFileName:string[20];
  DoContinue:boolean;

  Procedure BiMaxiNet;

IMPLEMENTATION

  uses CRT;

  var 
  ii,jj,cap:longint;
  i,k,kk,z,num,col,imax,jmax,capacity:longint;
  num_nodes,num_arcs:longint;
  source,sink:longint;
  found:boolean;
  unit_cap:boolean;
  netres:text;
  delay1:integer;
  IOCode:integer;


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
      writeln('    BIPARTITE MAX-FLOW NETWORK');
      writeln('           GENERATOR');
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
      writeln('  Try the network generation out by');
      writeln('  following the instructions.');
      write('  Do you want to continue (Y/N) ');
      ReadResponse;
      clrscr;
    End;

  Procedure Banner2;
    Begin
      clrscr;
      writeln('  A BIPARTITE MAX-FLOW NETWORK will');
      writeln('  be generated to file ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The sink node is node   ',sink:8);
      writeln('  The sink node is numbered with the ');
      writeln('  largest node number. ');
      writeln('  The network is given as a list of');
      writeln('  lines: comment lines, problem line,');
      writeln('  node lines, arc lines as follows:');
      writeln('  c This is a comment line.');
      writeln('  c The "p" line gives the problem');
      writeln('  c TYPE, number of nodes, and ');
      writeln('  c number of arcs. Two "n" lines');
      write('  Please WAIT !');
      Delay(delay1);
      clrscr;
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
      writeln;
      writeln;
      writeln;
      writeln;
      write('  Please WAIT !');
      Delay(delay1);
      clrscr;
      writeln('  Please WAIT !');
      writeln;
      writeln(netres,'c Bipartite Network for Max-Flow');
      if unit_cap then
        writeln(netres,'c with UNIT capacities on bipartite arcs')
      else
        writeln(netres,'c with RANDOM capacities');
      writeln(netres,'p max',num_nodes:10,num_arcs:10);
      writeln(netres,'n',source:14,'  s');
      writeln(netres,'n',sink:14,'  t');
    End;

  Procedure Banner3;
    Begin
      writeln;
      writeln('  The network is completed. The data');
      writeln('  file is an ASCII file, and can be');
      writeln('  accessed with your editor. You may');
      writeln('  print the file by first returning');
      writeln('  to DOS and then using the DOS');
      writeln('  command PRINT ',OutFileName,'.max.');
      writeln;
      write('  PRESS ENTER TO RETURN TO MAIN MENU. ');
      readln;
    End;

  Procedure UserValues;
    Procedure SourceNodesRead;
    Begin
      repeat
        repeat
          {$I-}
          readln(imax);
          {$I+}
          IOCode:=IOResult;
          if IOCode<>0 then write('  Try again ! Number of nodes: ');
        until IOCode=0;
        if (imax<1) then
        write('  Try again. Need at least one node: ');
      until (imax>=1);
    End;

    Procedure SinkNodesRead;
    Begin
      repeat
        repeat
          {$I-}
          readln(jmax);
          {$I+}
          IOCode:=IOResult;
          if IOCode<>0 then write('  Try again ! Number of nodes: ');
        until IOCode=0;
        if (jmax<1) then
        write('  Try again. Need at least one node: ');
      until (jmax>=1);
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
      unit_cap:=false;
      writeln('  Type the number of nodes on');
      write('  the source side: ');
      SourceNodesRead;
      writeln('  Type the number of nodes on the');
      write('  sink side: ');
      SinkNodesRead;
      clrscr;
      writeln('  You have two choices for arc');
      writeln('  capacities: either unit capacities');
      writeln('  or random capacities.');
      writeln('  Do you want unit flow capacities');
      write('  on bipartite arcs (Y/N) ');
      ReadResponse;
      if (ch='y') or (ch='Y') then unit_cap:=true;
      if unit_cap then
        Begin
          writeln('  Give an upper bound for the ');
          writeln('  random CAPACITY for arcs out');
          write('  of the source and into the sink: ');
          CapacityRead;
        End
      else
        Begin
          writeln('  Give an upper bound for the');
          write('  random CAPACITY: ');
          CapacityRead;
        End;
      num_nodes:=imax+jmax+2;
      num_arcs:=imax+jmax+imax*jmax;
      source:=1;
      sink:=num_nodes;
    End;

Procedure BiMaxiNet;
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
      Banner2;
      BipartiteNet;
      close(netres);
      Banner3;
    End
    else DoContinue:=false;
  End;

End.
