{
 TRANSIT GRID NETWORK GENERATOR FOR MAX-FLOW/
 TWO-WAY SYSTEM           

 Copyright (C) (1990,1991) Gary R. Waissi
 University of Michigan-Dearborn
 School of Management, 113FOB
 Dearborn, MI 48128
 See file README1.TXT for detail.
}
{$O+,F+}
UNIT Tr2Maxi;

INTERFACE

  var ch:char;
      OutFileName:string[20];
      DoContinue:boolean;

  Procedure Tr2MaxiNet;

IMPLEMENTATION

  uses CRT;

  var ii,jj,kk,zz,cap:longint;
      i,z,k,num,col,imax,zmax,colmax,capacity:longint;
      num_nodes,num_arcs:longint;
      source,sink:longint;
      netres:text;
      delay1:integer;
      IOCode:integer;
  
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
      writeln('    TWO-WAY TRANSIT GRID NETWORK');
      writeln('       GENERATOR FOR MAX-FLOW');
      writeln;
      writeln('  This program generates a TWO-WAY');
      writeln('  TRANSIT GRID network for MAX-FLOW');
      writeln('  into a user file. The network ');
      writeln('  resembles a two-way city street');
      writeln('  network.');
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
    End;

  Procedure Banner3;
    Begin
      clrscr;
      writeln('  A TWO-WAY TRANSIT GRID NETWORK for');
      writeln('  MAX-FLOW will be generated to file');
      writeln('    ',OutFileName,'.max.');
      writeln('  The number of nodes is  ',num_nodes:8);
      writeln('  The number of arcs is   ',num_arcs:8);
      writeln('  The source node is node ',source:8);
      writeln('  The sink node is node   ',sink:8);
      writeln('  The sink node is numbered with the');
      writeln('  largest node number.');
      writeln;
      writeln('  The network is given as a list of');
      writeln('  lines: comment lines, problem line,');
      writeln('  node lines, arc lines as follows:');
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
      write('  Please WAIT!');
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
    Procedure NodesRead;
    Begin
      repeat
        repeat
          {$I-}
          readln(imax);
          {$I+}
          IOCode:=IOResult;
          if IOCode<>0 then write('  Try again ! Number of nodes: ');
        until IOCode=0;
        if (imax<4) then
        Begin
          writeln('  Try again. Need at least four');
          writeln('  nodes in the smallest grid ');
          write('  network. Number of nodes: ');
        End;
      until (imax>=4);
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

    Begin
      writeln('  How many nodes are in the network ?');
      write('  (4, 9, 16, 25,..., 400, 625,...): ');
      NodesRead;
      colmax:=round(sqrt(imax));
      zmax:=2*(imax-colmax);
      writeln('  What is the upper bound for arc flow');
      write('  CAPACITY: ');
      CapacityRead;
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

  Procedure Tr2MaxiNet;
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
      Banner2;
      UserValues;
      TransitTwoNet;
      close(netres);
      Banner4;
    End
    else DoContinue:=false;
  End;

End.
