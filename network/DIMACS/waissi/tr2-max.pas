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
