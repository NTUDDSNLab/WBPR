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
