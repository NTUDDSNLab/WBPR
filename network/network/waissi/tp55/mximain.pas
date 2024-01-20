{---------------------------------------------------}
{ UNIT MxiMain made from                            }
{ PROGRAM NetgenMaxflow(INPUT,OUTPUT);              }
{ Version updated January 18,1991                   }
{ This is the MAIN PROGRAM of the MAX-FLOW SOLVER   }
{}
{    Gary R. Waissi, Copyright (C) (1990,1991)      }
{}
{---------------------------------------------------}
{ Uses DIMACS data files.                           }
{ This version can solve fairly large networks.     }
{ Limiting factors are: a single array that is used }
{ to store the level network, and the dynamic       }
{ variables are stored in a heap of limited size.   }
{ Please change this ARRAY to a dynamic data        }
{ structure                                         }
{                                                   }
{   Level              :ARRAY[0..nmax] of integer;  }
{                                                   }
{ for example alinked list (I RUN OUT OF TIME AS I  }
{ WAS CONVERTING THE DATA STRUCTURES TO TREES).     }
{ The other limiting factor, the HEAP SIZE, can be  }
{ controlled using the M compiler option in Turbo   }
{ Pascal 5.5. The dynamic variables are stored in   }
{ a heap, and the heap fills up very fast.          }
{}
{ POSSIBLE IMPROVEMENTS:                            }
{ I am using two balanced trees: a node-tree and    }
{ an arc-tree to store my network data. However,    }
{ this can be converted to one tree of nodes, and   }
{ then have two lists of arcs attached to each node,}
{ one list for entering arcs and one list for       }
{ leaving arcs.                                     }
{ Also, my program does not take advantage of fast  }
{ tree searching, i.e. search a node (or arcs) by   }
{ some index or address. I always search the tree   }
{ starting from the left sub-tree until I find the  }
{ element I'm looking for. This was easy to program,}
{ but it is not efficient.                          }

UNIT MxiMain;
{$M 65000,0,512000}

{  Version written on 8/28/90           }
{}
{ NETWORK GENERATOR 1.200               }
{ NBF ALGORITHM 1.200                   }
{       Updated July 1990               }
{ Uses trees as data structures to      }
{ store the original network arcs,      }
{ in an ArcTree, and the original       }
{ network nodes in a NodeTree.          }
{}
{     Algorithms and programs by        }
{          Gary R. Waissi               }
{       School of Management            }
{   Department of Decision Science      }
{      Information Systems and          }
{      Operations Management            }
{  University of Michigan - Dearborn    }
{             1986-1989                 }
{}
{ USES MIN AD(i) IN NODE SELECTION      }
{ Several other versions are available. }
{}

INTERFACE

  uses DOS,CRT,Explain,MxiSub1,MxiSub2,MxiSub3,MxiSub4;
  Procedure FindMaximalFlow;

IMPLEMENTATION

  Procedure Banner;
    begin
      if Intermediate then
        begin
          writeln;
          writeln('Source node is     ',so:4);
          writeln('Sink node is       ',T:4);
          writeln('Number of nodes is ',imax:4);
          writeln('Number of arcs is  ',jmax:4);
          writeln('Intermediate flow values on arcs.');
          writeln;
        end
    end;

  Procedure Heading;
    begin
      WayToGo;
      ClrScr;
      writeln;
      writeln('    Arc   From  To   Flow   Kappa');
      writeln;
    end;

  Procedure WriteOut;
    begin
      if not WriteResults then
        writeln(J:6,Arc^.TailNode:6,Arc^.HeadNode:6,
                  Arc^.ArcFlow:6,Arc^.ArcCapacity:6);
      if WriteResults then
        writeln(Netres,J:6,Arc^.TailNode:6,Arc^.HeadNode:6,
                           Arc^.ArcFlow:6,Arc^.ArcCapacity:6)
      else if not WriteResults then
      begin
        Line:=Line+1;
        if (Line=10) then
        begin
          WayToGo;
          ClrScr;
          writeln('    Flow values on arcs');
          writeln('    Arc   From  To   Flow   Cap');
          writeln;
        end;
      end;
    end;

  Procedure Results1;
    begin
      Banner;
      writeln('    Initial flow values on arcs');
      writeln('    Arc   From  To   Flow   Cap');
      Line:=0;
      for J:=1 TO jmax DO
      begin
        LocateArc(ArcTreeRoot,J);
        WriteOut;
      end;
    end;

  Procedure Results2;
    begin
      if not WriteResults then
        begin
          ClrScr;
          writeln('    Flow values on arcs');
          writeln('    Arc   From  To   Flow   Cap');
          writeln;
        end
      else if WriteResults then
        begin
          ClrScr;
          writeln('SAVING FILE. PLEASE WAIT!');
          writeln(Netres,'Source node     ',so:4,
                         ' Sink node      ',T:4);
          writeln(Netres,'Number of nodes ',imax:4,
                         ' Number of arcs ',jmax:4);
          writeln(Netres,'Maximum flow is: ',MaxFlow);
          writeln(Netres,'Total number of network ',
                         'generations: ',net-1);
          writeln(Netres);
          writeln(Netres,'    Final flow values on arcs');
          writeln(Netres,'    Arc   From  To   Flow   Cap');
        end;
      Line:=0;
      for J:=1 TO jmax DO
      begin
        LocateArc(ArcTreeRoot,J);
        if not (Arc^.ArcFlow=0) then WriteOut;
      end;
      if Writeresults then
      begin
        writeln(Netres);
        writeln(Netres,'Note: The flow is zero on all ');
        writeln(Netres,'arcs not listed here.');
      end
      else if not WriteResults then
      begin
        writeln;
        writeln('Note: The flow is zero on all arcs not ');
        writeln('listed above.');
        GoOn;
      end;
    end;

  Procedure ArcsEplus;
    var FFP:integer;
    begin
      Line:=0;
      writeln;
      writeln('The set E+ contains the following arcs:');
      writeln('    Arc   From  To   Flow   Kappa');
      writeln;
      for J:=1 TO jmax DO
      begin
        LocateArc(ArcTreeRoot,J);
        if (Arc^.EplusSet) then
        begin
          FFP:=Arc^.ArcFlow-Arc^.TemporaryFlow;
          if not (FFP=0) then
            begin
              writeln(J:6,Arc^.TailNode:6,Arc^.HeadNode:6,FFP:6,
                      Arc^.Kappa:6);
              Line:=Line+1;
              if (Line=10) then Heading;
            end;
        end;
      end;
      GoOn;
    end;

  Procedure ArcsEminus;
    var FFM:integer;
    begin
      Line:=0;
      writeln;
      writeln('The set E- contains the following arcs:');
      writeln('    Arc   From  To   Flow   Kappa');
      writeln;
      for J:=1 TO jmax DO
      begin
        LocateArc(ArcTreeRoot,J);
        if (Arc^.EminusSet) then
        begin
          FFM:=Arc^.TemporaryFlow-Arc^.ArcFlow;
          if not (FFM=0) then
            begin
              writeln(J:6,Arc^.TailNode:6,Arc^.HeadNode:6,FFM:6,
                      Arc^.Kappa:6);
              Line:=Line+1;
              if (Line=10) then Heading;
            end;
        end;
      end;
      GoOn;
    end;

  Procedure GetMaxFlow;
    begin
      for J:=1 TO jmax DO
        begin
          LocateArc(ArcTreeRoot,J);
          if (so=Arc^.TailNode) then
            MaxFlow:=MaxFlow+Arc^.ArcFlow
          else if (so=Arc^.HeadNode) then
            MaxFlow:=MaxFlow-Arc^.ArcFlow;
        end;
    end;

  Procedure Levels;
    begin    (* levels *)
      if (inp=false) then
        begin
          clrscr;
          writeln;
          writeln('This is the Acyclic Network no.: ',net);
          Banner;
          GetMaxFlow;
          if (inp=false) and (Intermediate=true) then
            begin
              writeln('Maximal flow is: ',MaxFlow);
              GoOn;
            end
          else if (inp=false) and (Intermediate=false) then
            begin
              writeln('Maximum flow is: ',MaxFlow);
              GoOn;
            end;
          if (inp=false) and (Intermediate=true) then
            Begin
              NetworkLevels;
              ArcsEplus;ArcsEminus;
            End;
          Results2;
        end;
    end;

  Procedure FindCutSet;  
    var POS,II,Line1:integer;
    Procedure WriteToScreenOrFile;
      begin
        LocateNode(NodeTreeRoot,I);
        if (Node^.ScannedSet) then
        begin
          if (POS<6) then
            begin
              if not WriteResults then write(I:5)
              else if WriteResults then write(Netres,I:5);
              POS:=POS+1;
              if (POS=6) then
              begin
                if not WriteResults then writeln
                else if WriteResults then writeln(Netres);
                POS:=1;
                Line1:=Line1+1;
                if (Line1=10) and not WriteResults then
                  begin
                    WayToGo;
                    ClrScr;
                    writeln('    More Nodes of the set X');
                    writeln;
                    Line1:=0;
                  end;
              end;
            end;
        end;
      end;

    Procedure CutSetNodes;
      begin
        if not WriteResults then
        begin
          writeln('The set X of the partition  X,X-bar ');
          writeln('contains following nodes');
          writeln('{');
        end
        else if WriteResults then
          begin
            writeln(Netres);
            writeln(Netres,'The set X of the partition ');
            writeln(Netres,'X,X-bar contains following nodes');
            writeln(Netres,'{');
          end;
        Line1:=0;
        for I:=1 TO imax DO WriteToScreenOrFile;
        if not WriteResults then
        begin
          writeln; writeln('}'); writeln;
        end
        else if WriteResults then
          begin
            writeln(Netres);
            writeln(Netres,'}');
            writeln(Netres);
          end;
      end;

    Procedure CutSetArcs;
      begin
        if not WriteResults then
        begin
          writeln('The cutset contains following arcs');
          writeln('    Arc   From  To   Flow   Cap');
        end
        else if WriteResults then
          begin
            writeln(Netres,'    The cutset contains following arcs');
            writeln(Netres,'    Arc   From  To   Flow   Cap');
          end;
        for I:=1 TO imax DO
        begin
          LocateNode(NodeTreeRoot,I);
          if (Node^.ScannedSet) then
            begin
              for J:=1 TO jmax DO
              begin
                LocateArc(ArcTreeRoot,J);
                if (I=Arc^.TailNode) then
                begin
                  II:=Arc^.HeadNode;
                  LocateNode(NodeTreeRoot,II);
                  if not Node^.ScannedSet then WriteOut
                end
                else if (I=Arc^.HeadNode) then
                begin
                  II:=Arc^.TailNode;
                  LocateNode(NodeTreeRoot,II);
                  if not Node^.ScannedSet then WriteOut;
                end;
              end;
            end;
        end;
        if not WriteResults then GoOn;
      end;

    begin {FindCutSet}
      Line:=0;
      POS:=1;
      if not WriteResults then ClrScr;
      CutSetNodes;
      if not WriteResults then GoOn;
      CutSetArcs;
    end;

  Procedure GenerateOutput;
    begin
      SaveData:=true;
      net:=0; inp:=false;
      ClrScr;
      writeln('Do you want to look at all Intermediate');
      write('results',' (Y/N) ');
      readln(ch);
      if (ch='Y') or (ch='y') then IntermediateResults:=true;
      if not IntermediateResults then
        begin
          writeln('Source node     ',so:4);
          writeln('Sink node       ',T:4);
          writeln('Number of nodes ',imax:4);
          writeln('Number of arcs  ',jmax:4);
          writeln('Network Generation: PHASE  I');
          writeln('Maximal Flow      : PHASE II ');
          writeln('Network     Max-Flow    Value');
          writeln('  Nr       Iterations  of Flow');
        end;
      begin
        repeat
          if IntermediateResults then
          writeln('Initializing');
          Initialize;
          if IntermediateResults then
          writeln('Preparing Data');
          Prepare;
          net:=net+1;
          if not IntermediateResults then write(net:4)
          else if IntermediateResults then
          writeln('Generating Network');
          GenerateNetwork;
          cmax:=C;
          if not Terminate then MaximalFlow;
          if not IntermediateResults then
            begin
              GetMaxFlow;
              writeln(nr:12,MaxFlow:12);
              nr:=0;
            end;
          if IntermediateResults then Levels;
        until Terminate;
        if not IntermediateResults and Terminate then
          begin
            writeln('The maximum value flow:              ',
                     MaxFlow);
            writeln('Total number of network generations: ',
                     net-1);
            write('Do you want to see the CUTSET',' (Y/N) ');
            readln(ch);
            if (ch='Y') or (ch='y') then FindCutset;
          end;
      end;
      Values:=false;
    end;

  Procedure InitMax;
    begin
      OKset:=['1','2','3','4','5','6','7','8'];
      ReadyToLeave:=false;
    end;

  Procedure DisplayInput;
    begin
      if (inp=true) then
      begin 
        ClrScr;
        Results1;
      end;
      GoOn;
    end;

  Procedure RetrieveData;
    var maximal:boolean;
        tryit:integer;
    begin
      maximal:=true;
      tryit:=0;
      SaveData:=false;
      Values:=true; inp:=true; Intermediate:=true;
      IntermediateResults:=false;
      Mainmenu:=false;
      x1:=0;
      {$I-}
      repeat
        if not xample then
        begin
          write('    Enter input file name: ');
          readln(InFileName);
        end
        else InFileName:='Example.max';
        ASSIGN(InputData,InFileName);
        RESET(InputData);
        IOCode:=Ioresult;
        if IOCode<>0 then
          begin
            writeln('File ',InFileName,' does not exist.');
            writeln('Try again.');
            write('Do you want to stop. (Y/N) ');
            readln(ch);
            if (ch ='Y') or (ch='y') then
              Mainmenu:=true;
          end;
      until (IOCode=0) or Mainmenu;
      {$I+}
      begin
        {$I-}
        repeat
          repeat
            read(InputData,ch);
            if (ch='c') then
            begin
              readln(InputData);
              tryit:=tryit+1;
            end
          until (ch<>'c');
          if (ch='p') then
          begin
            read(InputData);
            repeat
              read(InputData,ch);
              if (ch=' ') then read(InputData,ch);
            until not (ch=' ');

            if (ch='m') then
            begin
              read(InputData,ch);
              if (ch='a') then read(InputData,ch)
              else maximal:=false;
              if (ch='x') then maximal:=true;
            end
            else maximal:=false;

            If not maximal then
            begin
              writeln('This program solves max-flow problems only.');
              mainmenu:=true;
            end
            else if maximal then
              readln(InputData,imax,jmax);
          end
          else if not (ch='p') then
          begin
            writeln('Expecting a problem line.');
            writeln('Check your data file.');
            write('Press ENTER to return to MAINMENU ');
            readln;
            Mainmenu:=true;
          end;
          if (tryit>=10) then
            begin
              writeln('No problem line found among the 10');
              writeln('first lines. Press ENTER to return');
              write('to MAINMENU. ');
              readln;
              Mainmenu:=true;
            end;
        until maximal or Mainmenu;
        {$I+}
        IOCode:=Ioresult;
        if (IOCode<>0) then
            writeln('Check your data file.');
        if not mainmenu and maximal then
        begin
          {$I-}
          repeat
            read(InputData,ch);
            if (ch<>'n') then readln(InputData)
            else if (ch='n') then
            begin
              read(InputData,so);
              read(InputData,ch);
              if (ch=' ') then
               repeat
                 read(InputData,ch);
               until not (ch=' ');
            end;
          until (ch='s');
          readln(InputData);
          repeat
            read(InputData,ch);
            if (ch<>'n') then readln(InputData)
            else
            begin
              read(InputData,T);
              read(InputData,ch);
              if (ch=' ') then
                repeat
                  read(InputData,ch);
                until not (ch=' ');
            end;
          until (ch='t');
          readln(InputData);
          {$I+}
          IOCode:=Ioresult;
          if (IOCode<>0) or (so<>1) and (T<>imax) then
            begin
              writeln('Check your data file.');
              writeln('The source node must be 1 and the ',
                      'sink node ',imax);
            end;
          xx1:=0;
          NodeTreeRoot:=NodeTree(imax);
          ArcTreeRoot:=ArcTree(jmax);
        end;
      end;
    end;

  Procedure Example;
  begin
    xample:=true;
    ClrScr;
    writeln('Looking for file EXAMPLE.MAX !');
    RetrieveData;
    if not Mainmenu then
    begin
      writeln('Press ENTER and select MENU item 3 ');
      writeln('to look at the example data, or ');
      writeln('select MENU item 5 to SOLVE the ');
      write('example problem. ');
      readln;
    end;
    xample:=false;
  end;

  Procedure SaveResults;
    begin
      WriteResults:=true;
      {$I-}
      repeat
        write('    Enter name of output file: ');
        readln(OutFileName);
        ASSIGN(Netres,OutFileName);
        REWRITE(Netres);
        IOCode:=Ioresult;
        if (IOCode<>0) then
        begin
          writeln('Output file name must be a legal DOS ');
          writeln('filename. F.ex. myfile.txt');
        end;
      until IOCode=0;
      {$I+}
      Results2;
      FindCutset;
      CLOSE(Netres);
      writeln; write('Data written.');
      WriteResults:=false;
      GoOn;
    end;

Procedure WriteMenu;
begin
  ClrScr;
  writeln('  ----------------------------------------');
  writeln('          Network Generator  v. 1.0');
  writeln('          Max-Flow Algorithm v. 2.4');
  writeln('             Uses MIN(AD(i))          ');
  writeln('      USES TREES AS DATA STRUCTURES');
  writeln('             ***  MENU  ***           ');
  writeln('  ----------------------------------------');
  writeln('    1) Explain this program');
  Delay(100);
  writeln('    2) Show example case');
  Delay(100);
  writeln('    3) Display input');
  Delay(100);
  writeln('    4) Retrieve data from a file');
  Delay(100);
  writeln('    5) Solve');
  Delay(100);
  writeln('    6) Save final results to a file');
  Delay(100);
  writeln('    7) Return to MAIN MENU');
end;

Procedure FindMaximalFlow;
  begin
    xample:=false;
    SaveData:=false;
    Values:=false; Intermediate:=false;
    IntermediateResults:=false;
    WriteResults:=false;
    InitMax;
    repeat
      WriteMenu;
      writeln; writeln;
      write('    Select by number and press RETURN ');
      readln(ch);
      if ch IN OKset then
        CASE ch of
        '1': ExplainProgram;
        '2': Example;
        '3': DisplayInput;
        '4': RetrieveData;
        '5': if Values then GenerateOutput
             else
               begin
                 writeln('Must input values first. ');
                 writeln('Use MENU items 2 or 4. ');
                 GoOn;
               end;
        '6': if SaveData then SaveResults
             else
               begin
                 writeln('Must SOLVE the problem first. ');
                 GoOn;
               end;
        '7': ReadyToLeave:=true;
        end;
    until ReadyToLeave;
  end;

end.
