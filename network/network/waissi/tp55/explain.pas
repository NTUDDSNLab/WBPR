UNIT explain;

INTERFACE

   var ch:char;

PROCEDURE InitPage(PAGE,LASTPAGE:INTEGER);
PROCEDURE ExplainProgram;

IMPLEMENTATION
   uses CRT;

Procedure InitPage(PAGE,LASTPAGE:integer);
begin
  ClrScr;
  writeln('PROGRAM EXPLANATION     PAGE ',PAGE,' OF ',LASTPAGE);
  write('PRESS:');
  highvideo;
  write('    C');
  normvideo;
  write(' to CONTINUE');
  highvideo;
  write('    B');
  normvideo;
  writeln(' to BACK UP ');
  writeln('----------------------------------------');
end;

Procedure ExplainProgram;
  var PG : integer;
  Procedure PAGE1;
  begin
writeln('Algorithms and programs developed by');
writeln('Gary R. Waissi, University of Michigan -');
writeln('Dearborn, School of Management, Dearborn');
writeln('Michigan, 48128.');
writeln('This program calculates single commodity');
writeln('maximum value flows in directed networks');
writeln('using a two phase approach. In the first');
writeln('phase a new acyclic network generator is');
writeln('applied. In the second phase a new maximal');
writeln('flow algorithm is applied to the generated');
writeln('acyclic network. For maximum value flow');
writeln('the two phases are repeated until the ');
writeln('network generation phase fails indicating');
writeln('that the maximum value flow has been found.');
  end;
  Procedure PAGE2;
  begin
writeln('The program uses two dynamic data');
writeln('structures (balanced trees) to store and');
writeln('update network data: a node tree and an');
writeln('arc tree. Data can be read from (and ');
writeln('written to) a user file. Be sure, when ');
writeln('using your own files, that the data file');
writeln('does not contain any empty lines, and that');
writeln('the nodes are numbered consecutively with ');
writeln('no number missing in-between. The numbering');
writeln('of nodes may be done in any order, but the');
writeln('source node must be assigned number 1, and');
writeln('the sink node the largest number. Otherwise');
writeln('the network data file must correspond to');
writeln('the DIMACS specifications.');
  end;
  Procedure PAGE3;
  begin
writeln('FORMAT OF A NETWORK DATA FILE:');
writeln('Line c:  Comment line');
writeln('Line p:  Problem line. Type of the problem,');
writeln('         the number of nodes and arcs.');
writeln('Line n:  Node line. Source node number, and');
writeln('         a node designator s.');
writeln('Line n:  Node line. Sink node number, and');
writeln('         a node designator t.');
writeln('Line a:  Arc line. Three numbers/line:');
writeln('         Arc Tail-Arc-Head-Flow Capacity');
writeln('Try to create a data file and read it in.');
writeln('Use the data given on the next page. Your');
writeln('results should correspond to those of the');
writeln('example case of menu item 2.');
  end;
  Procedure PAGE4;
  begin
writeln('c EXAMPLE NETWORK DATA FILE');
writeln('p max  7      9 ');
writeln('n      1      s ');
writeln('n      7      t ');
writeln('a      1      3     13');
writeln('a      1      4      8');
writeln('a      3      2      2');
writeln('a      3      4      7');
writeln('a      2      5      6');
writeln('a      4      5      4');
writeln('a      2      6     10');
writeln('a      5      7      9');
writeln('a      6      7      6');
  end;

begin
  PG := 1;
  repeat
    InitPage(PG,4);
    CASE PG OF
      1: PAGE1; 2: PAGE2; 3: PAGE3; 4: PAGE4;
    end;
    GotoXY(8,2);
    readln(ch);
    if (ch = 'C') or (ch = 'c') then PG := PG + 1
    else if (ch = 'B') or (ch = 'b') then PG := PG - 1
  until (ch = 'E') or (ch = 'e') or (PG=5) or (PG=0)
end;

end.
