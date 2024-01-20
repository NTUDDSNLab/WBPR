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
