{---------------------------------------------------}
{ PROGRAM NetgenMaxflow(INPUT,OUTPUT);              }
{ Version updated January 18,1991                   }
{ This version is used as a core to make the        }
{ version for the package that contains the         }
{ max-flow generators.                              }
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

PROGRAM NetgenMaxflow(INPUT,OUTPUT);
{$M 65000,0,512000}

(**************************************)
(*  Version developed on 8/28/90      *)
(*                                    *)
(* NETWORK GENERATOR 1.200            *)
(* NBF ALGORITHM 1.200     July 1990  *)
(* Uses trees as data structures to   *)
(* store the original network arcs,   *)
(* in an ArcTree, and the original    *)
(* network nodes in a NodeTree.       *)
(*                                    *)
(*     Algorithms and program by      *)
(*          Gary R. Waissi            *)
(*       School of Management         *)
(*   Department of Decision Science   *)
(*      Information Systems and       *)
(*      Operations Management         *)
(*  University of Michigan - Dearborn *)
(*             June 1990              *)
(*                                    *)
(* USES MIN AD(i) IN NODE SELECTION   *)
(**************************************)
  uses DOS,CRT;

  const
    nmax = 32000;
    { change the data structure to a linked }
    { list  to eliminate the limitation:    }
    { Level :array[0..nmax] of integer;     }

  type

  NodeTreeType = ^NodeInfo;
  NodeInfo     =
    record
      NodeNumber       :integer;
      ia               :integer;
      ind              :integer;
      ad               :integer;
      BlockingStatus   :integer;
      FlowIn           :integer;
      FlowOut          :integer;
      Alpha            :integer;
      Beta             :integer;
      Rho              :integer;
      ActiveSet        :boolean;
      EligibleSet      :boolean;
      ScannedSet       :boolean;
      DnodesSet        :boolean;
      DelNodesSet      :boolean;
      LeftNodeTree,RightNodeTree:NodeTreeType;
    end;

  ArcTreeType  = ^ArcRecord;
  ArcRecord    =
    record
      ArcNumber        :integer;
      TailNode         :integer;
      HeadNode         :integer;
      ArcFlow          :integer;
      ArcCapacity      :integer;
      Admissible       :integer;
      Kappa            :integer;
      TemporaryFlow    :integer;
      TemporaryCapacity:integer;
      EplusSet         :boolean;
      EminusSet        :boolean;
      DarcsSet         :boolean;
      LeftArcTree,RightArcTree:ArcTreeType;
    end;

  var
    Node               :NodeTreeType;
    Arc                :ArcTreeType;
    NodeTreeRoot       :NodeTreeType;
    ArcTreeRoot        :ArcTreeType;

    IOCode             :integer;
    
    nr,net             :integer;
    imax,jmax,cmax     :integer;
    C,I,J,K,L,P,Y,T    :integer;
    yy,so,z,no,nod     :integer;
    G,f1,Excess        :integer;
    MaxFlow            :integer;
    Line               :integer;

    ActiveNodes,EligibleNodes,ScannedNodes:integer;
    DelNodes,DNodes    :integer;

    EplusArcs,EminusArcs,Darcs:integer;

    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10:integer;
    x11,x12,x13,x14,x15:boolean;
    xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8,xx9:integer;
    xx10,xx11,xx12:boolean;

    ch                 :CHAR;

    ZeroFlow           :boolean;
    Values,inp         :boolean;
    Intermediate       :boolean;
    ReadyToLeave       :boolean;
    IntermediateResults:boolean;
    SaveData           :boolean;
    WriteResults       :boolean;
    Found,Done         :boolean;
    Terminate          :boolean;
    xample             :boolean;
    Mainmenu           :boolean;

    OKset              :set of CHAR;

    Level              :array[0..nmax] of integer;

    OutFileName        :string[10];
    Netres             :text;
    InFileName         :string[20];
    InputData          :text;
    problem            :string[5];

Function NodeTree(n:integer):NodeTreeType;
  var NewNode:NodeTreeType;
      nel,nar:integer;
begin
  if n=0 then NodeTree:=nil else
  begin
    nel:=n DIV 2; nar:=n-nel-1;
    x1:=x1+1;
    x2:=0;x3:=0;x4:=0;x5:=0;x6:=0;x7:=0;x8:=0;x9:=0;x10:=0;
    x11:=false;x12:=false;x13:=false;x14:=false;x15:=false;
    if (x1=1) then x11:=true;
    NEW(NewNode);
    WITH NewNode^ DO
      begin
        NodeNumber    :=x1;
        ia            :=x2;
        ind           :=x3;
        ad            :=x4;
        BlockingStatus:=x5;
        FlowIn        :=x6;
        FlowOut       :=x7;
        Alpha         :=x8;
        Beta          :=x9;
        Rho           :=x10;
        ActiveSet     :=x11;
        EligibleSet   :=x12;
        ScannedSet    :=x13;
        DnodesSet     :=x14;
        DelNodesSet   :=x15;
        LeftNodeTree  :=NodeTree(nel);
        RightNodeTree :=NodeTree(nar);
      end;
    NodeTree:=NewNode;
  end;
end;

Function ArcTree(nn:integer):ArcTreeType;
  var NewArc:ArcTreeType;
      nll,nrr:integer;
begin
  if nn=0 then ArcTree:=nil else
  begin
    nll:=nn DIV 2; nrr:=nn-nll-1;
    {$I-}
    repeat
      read(InputData,ch);
      if (ch<>'a') then readln
      else
      begin
        xx1:=xx1+1;
        readln(InputData,xx2,xx3,xx5);
      end;
    until (ch='a');
    {$I+}
    IOCode:=Ioresult;
    if (IOCode<>0) then
    writeln('Check data file at arc number ',xx1);
    xx4:=0;xx6:=0;xx7:=0;xx8:=0;xx9:=0;
    xx10:=false;xx11:=false;xx12:=false;
    NEW(NewArc);
    WITH NewArc^ DO
      begin
        ArcNumber        :=xx1;
        TailNode         :=xx2;
        HeadNode         :=xx3;
        ArcFlow          :=xx4;
        ArcCapacity      :=xx5;
        Admissible       :=xx6;
        Kappa            :=xx7;
        TemporaryFlow    :=xx8;
        TemporaryCapacity:=xx9;
        EplusSet         :=xx10;
        EminusSet        :=xx11;
        DarcsSet         :=xx12;
        LeftArcTree      :=ArcTree(nll);
        RightArcTree     :=ArcTree(nrr);
      end;
    ArcTree:=NewArc;
  end;
end;

Procedure LocateNode(t1:NodeTreeType; h1:integer);
  var Found1:boolean;
begin
  Found1:=false;
  while (t1<>nil) and not Found1 DO
  begin
    if t1^.NodeNumber=h1 then Found1:=true
    else if h1=t1^.LeftNodeTree^.NodeNumber
    then t1:=t1^.LeftNodeTree
    else if h1=t1^.RightNodeTree^.NodeNumber
    then t1:=t1^.RightNodeTree
    else if h1<t1^.RightNodeTree^.NodeNumber
    then t1:=t1^.LeftNodeTree
    else if h1>t1^.RightNodeTree^.NodeNumber
    then t1:=t1^.RightNodeTree;
  end;
  if Found1 then Node:=t1;
end;

Procedure LocateArc(t2:ArcTreeType; h2:integer);
  var Found2:boolean;
begin
  Found2:=false;
  while (t2<>nil) and not Found2 DO
  begin
    if t2^.ArcNumber=h2 then Found2:=true
    else if h2=t2^.LeftArcTree^.ArcNumber
    then t2:=t2^.LeftArcTree
    else if h2=t2^.RightArcTree^.ArcNumber
    then t2:=t2^.RightArcTree
    else if h2<t2^.RightArcTree^.ArcNumber
    then t2:=t2^.LeftArcTree
    else if h2>t2^.RightArcTree^.ArcNumber
    then t2:=t2^.RightArcTree;
  end;
  if Found2 then Arc:=t2;
end;

Procedure PrintNodeTree(NodeTree1:NodeTreeType);
begin
  if NodeTree1<>nil then
  WITH NodeTree1^ DO
  begin
    PrintNodeTree(LeftNodeTree);
    writeln(NodeNumber:3,ia:5,ind:5,ad:5,BlockingStatus:5,
            FlowIn:5,FlowOut:5,Alpha:5,Beta:5,Rho:5);
    PrintNodeTree(RightNodeTree);
  end;
end;

Procedure PrintArcTree(ArcTree1:ArcTreeType);
begin
  if ArcTree1<>nil then
  WITH ArcTree1^ DO
  begin
    PrintArcTree(LeftArcTree);
    writeln(ArcNumber:3,TailNode:5,HeadNode:5,
            ArcFlow:5,ArcCapacity:5,Admissible:5,Kappa:5,
            TemporaryFlow:5,TemporaryCapacity:5);
    PrintArcTree(RightArcTree);
  end;
end;

Procedure WayToGo;
  begin
    writeln;
    write('To continue, press ENTER <ret> ');
    readln;
    ClrScr;
    Line:=0;
  end;

Procedure GoOn;
  begin
    writeln;
    write('To continue, press ENTER <ret> ');
    readln;
    ClrScr;
  end;

Procedure NetworkLevels;
  begin
    Line:=0;
    writeln;
    writeln('Levels of the Acyclic Network');
    writeln;
    C:=0;
    repeat
      begin
        write('Level ',C,': ');
        writeln(Level[C]:4)
      end;
      Line:=Line+1;
      if (Line=15) then WayToGo;
      C:=C+1;
    until (C=cmax);
    GoOn;
  end;

Procedure IncludeToActive;
  begin
    Node^.ActiveSet:=true;
    ActiveNodes:=ActiveNodes+1;
  end;

Procedure ExcludeFromActive;
  begin
    Node^.ActiveSet:=false;
    ActiveNodes:=ActiveNodes-1;
  end;

Procedure IncludeToEligible;
  begin
    Node^.EligibleSet:=true;
    EligibleNodes:=EligibleNodes+1;
  end;
  
Procedure ExcludeFromEligible;
  begin
    Node^.EligibleSet:=false;
    EligibleNodes:=EligibleNodes-1;
  end;

Procedure IncludeToScanned;
  begin
    Node^.ScannedSet:=true;
    ScannedNodes:=ScannedNodes+1;
  end;

Procedure IncludeToDarcs;
  begin
    Arc^.DarcsSet:=true;
    Darcs:=Darcs+1;
  end;

Procedure IncludeToEplusSet;
  begin
    Arc^.EplusSet:=true;
    EplusArcs:=EplusArcs+1;
  end;

Procedure ExcludeEplusSet;
  begin
    Arc^.EplusSet:=false;
    EplusArcs:=EplusArcs-1;
  end;

Procedure IncludeToEminusSet;
  begin
    Arc^.EminusSet:=true;
    EminusArcs:=EminusArcs+1;
  end;

Procedure ExcludeEminusSet;
  begin
    Arc^.EminusSet:=false;
    EminusArcs:=EminusArcs-1;
  end;

Procedure Initialize;
  begin
    K:=0; L:=0; Y:=0; yy:=0; Z:=0; no:=0; nod:=0;
    cmax:=0; MaxFlow:=0;
    C:=0; P:=1;
    ActiveNodes:=0;EligibleNodes:=0;ScannedNodes:=0;
    DelNodes:=0;Dnodes:=0;
    EplusArcs:=0;EminusArcs:=0;Darcs:=0;
    for I:=1 to imax DO
    begin
      LocateNode(NodeTreeRoot,I);
      Node^.FlowOut:=0; Node^.FlowIn:=0;
      Node^.Alpha:=0; Node^.Beta:=0; Node^.Rho:=0;
      Node^.ActiveSet  :=false;
      Node^.EligibleSet:=false;
      Node^.ScannedSet :=false;
      Node^.DnodesSet  :=false;
      Node^.DelNodesSet:=false;
    end;
    I:=1;
    LocateNode(NodeTreeRoot,I);
    IncludeToActive;
    for J:=1 to jmax DO
    begin
      LocateArc(ArcTreeRoot,J);
      Arc^.EplusSet  :=false;
      Arc^.EminusSet :=false;
      Arc^.DarcsSet  :=false;
    end;
    for I:=1 TO imax DO Level[I]:=0;
  end;

Procedure Prepare;
  Procedure admiss1;
    begin
      writeln('The flow on arc ',J,' ',Arc^.ArcFlow,
              ' is greater than');
      writeln('the arc capacity ',Arc^.ArcCapacity);
      writeln('Check your data!');
      Terminate:=true;
    end;
  Procedure admiss2;
    begin
      if (Arc^.HeadNode=so) or (Arc^.TailNode=T) then 
        IncludeToDarcs
      else Arc^.Admissible:=2;
    end;
  Procedure admiss3;
    begin
      if (Arc^.TailNode=so) or (Arc^.HeadNode=T) then
        IncludeToDarcs
      else Arc^.Admissible:=3;
     end;
  Procedure admiss4;
    begin
      if (Arc^.TailNode=so) then Arc^.Admissible:=2
      else if (Arc^.HeadNode=so) then Arc^.Admissible:=3
      else if (Arc^.HeadNode=T) then Arc^.Admissible:=2
      else if (Arc^.TailNode=T) then Arc^.Admissible:=3
      else Arc^.Admissible:=1;
    end;

  Procedure ForwardNodeNumbers;
    begin
      if (Arc^.Admissible=1) or (Arc^.Admissible=2) then
        Node^.ad:=Node^.ad+1
      else if (Arc^.Admissible=3) then Node^.ia:=Node^.ia+1;
    end;

  Procedure ReverseNodeNumbers;
    begin
      Node^.ind:=Node^.ind+1;
      if (Arc^.Admissible=1) or (Arc^.Admissible=3) then
        Node^.ad:=Node^.ad+1
      else if (Arc^.Admissible=2) then Node^.ia:=Node^.ia+1;
    end;

  begin  { Prepare }
    Terminate:=false;
    for J:=1 TO jmax DO
    begin
      LocateArc(ArcTreeRoot,J);
      if (Arc^.ArcFlow>Arc^.ArcCapacity) then admiss1
      else if (Arc^.ArcCapacity=0) then IncludeToDarcs
      else if (Arc^.ArcFlow=0) then admiss2
      else if (Arc^.ArcFlow=Arc^.ArcCapacity) then admiss3
      else if (Arc^.ArcFlow<Arc^.ArcCapacity) then admiss4;
      if not (Arc^.DarcsSet) then
      begin
        I:=Arc^.TailNode;
        LocateNode(NodeTreeRoot,I);
        ForwardNodeNumbers;
        I:=Arc^.HeadNode;
        LocateNode(NodeTreeRoot,I);
        ReverseNodeNumbers;
      end;
    end;
  end;   { Prepare }

Procedure GenerateNetwork;
  Procedure IncludeToLevel;
    begin
      Level[C]:=Y; C:=C+1;
      ExcludeFromActive;
    end;

  Procedure ScanNodeT;
    begin
      IncludeToScanned;
      Y:=T;
      IncludeToLevel;
    end;

  Procedure Update;
    Procedure Update1;
      begin
        LocateNode(NodeTreeRoot,K);
        Node^.ad:=Node^.ad-1;
        LocateNode(NodeTreeRoot,L);
        Node^.ad:=Node^.ad-1;
        Node^.ind:=Node^.ind-1;
      end;

    Procedure ForwardUpdate;
      Procedure ForwardUpdate2;
        begin
          LocateNode(NodeTreeRoot,K);
          Node^.ad:=Node^.ad-1;
          LocateNode(NodeTreeRoot,L);
          Node^.ia:=Node^.ia-1;
          Node^.ind:=Node^.ind-1;
        end;
      begin    (* ForwardUpdate *)
        if (Arc^.Admissible=1) or (Arc^.Admissible=2) then
          begin
            K:=Arc^.TailNode; L:=Arc^.HeadNode;
            if (Arc^.Admissible=1) then Update1
            else if (Arc^.Admissible=2) then ForwardUpdate2;
          end;
      end;

    Procedure BackwardUpdate;
      Procedure BackwardUpdate2;
        begin
          LocateNode(NodeTreeRoot,K);
          Node^.ia:=Node^.ia-1;
          LocateNode(NodeTreeRoot,L);
          Node^.ad:=Node^.ad-1;
          Node^.ind:=Node^.ind-1;
        end;

      begin    (* BackwardUpdate *)
        if (Arc^.Admissible=1) or (Arc^.Admissible=3) then
          begin
            K:=Arc^.TailNode; L:=Arc^.HeadNode;
            if (Arc^.Admissible=1) then Update1
            else if (Arc^.Admissible=3) then BackwardUpdate2;
          end;
      end;

    begin    (* Update *)
      if (Arc^.EplusSet) then ForwardUpdate
      else if (Arc^.EminusSet) then BackwardUpdate;
    end;
  
  Procedure Delete;
    var S,S1,ss:integer;
    (* THIS Procedure IS CHANGED TO MODIFY NODE SELECTION CRITERIA *)

    Procedure DetEligNode;
      begin
        if (EligibleNodes=0) then
          begin
            IncludeToEligible;
            ss:=Y;
            S:=Node^.ad;
            S1:=Node^.ia;
          end
        else if (S>Node^.ad) and (EligibleNodes<>0) then
          begin
            LocateNode(NodeTreeRoot,ss);
            ExcludeFromEligible;
            ss:=Y;
            LocateNode(NodeTreeRoot,ss);
            S:=Node^.ad;
            S1:=Node^.ia;
            IncludeToEligible;
          end;
        Found:=true;
      end;

    Procedure DeleteArc;
      Procedure DeleteForwardArc;
        begin
          L:=Arc^.HeadNode; S1:=S1-1;
          LocateNode(NodeTreeRoot,L);
          Node^.ad:=Node^.ad-1;
          Node^.ind:=Node^.ind-1;
          IncludeToDarcs;
        end;

      Procedure DeleteReverseArc;
        begin
          LocateNode(NodeTreeRoot,ss);
          Node^.ind:=Node^.ind-1;
          K:=Arc^.TailNode; S1:=S1-1;
          LocateNode(NodeTreeRoot,K);
          Node^.ad:=Node^.ad-1;
          IncludeToDarcs;
        end;

      begin    (* DeleteArc *)
        LocateArc(ArcTreeRoot,Z);
        if (ss=Arc^.TailNode) then
          begin
            if not Arc^.DarcsSet and
               not (Arc^.EplusSet or Arc^.EminusSet) then
               if (Arc^.Admissible=3) then DeleteForwardArc
               else if (ss=T) and (Arc^.Admissible=2) then
                 DeleteForwardArc;
          end
        else if (ss=Arc^.HeadNode) then
          begin
            if not Arc^.DarcsSet and
               not (Arc^.EplusSet or Arc^.EminusSet) then
               if (Arc^.Admissible=2) then DeleteReverseArc;
          end;
      end;

    begin                  (* Delete *)
      Found:=false;
      for Y:=1 TO imax DO
      begin
        LocateNode(NodeTreeRoot,Y);
        if (Node^.ActiveSet) and not (Y=T) then DetEligNode;
      end;
      if Found then
        begin
          for Z:=1 TO jmax DO DeleteArc;
          LocateNode(NodeTreeRoot,ss);
          Node^.ia:=S1;
          ExcludeFromActive;
          Level[C]:=ss; C:=C+1;
        end
      else
        begin
          LocateNode(NodeTreeRoot,T);
          if (Node^.ActiveSet) and (ActiveNodes=1) then
          begin
            IncludeToEligible;
            ss:=T;
            LocateNode(NodeTreeRoot,ss);
            S1:=Node^.ia;
            for Z:=1 TO jmax DO DeleteArc;
            LocateNode(NodeTreeRoot,T);
            Node^.ia:=S1;
            LocateNode(NodeTreeRoot,T);
            ScanNodeT;
            ExcludeFromEligible;
          end;
        end;
    end;

  Procedure NodeEligible;
    var tryother:boolean;

    Procedure FindEligibleNode;
      Procedure Eligible1;
        begin
          LocateNode(NodeTreeRoot,Y);
          if Node^.ActiveSet and (Node^.ia=0)
            and (Node^.ind=0) and not (Y=T) then
            begin
              IncludeToEligible;
              IncludeToLevel;
              tryother:=false;
            end;
        end;

      Procedure DetermineEligibleNode;
        Procedure SMALLEST;
          begin
            if (EligibleNodes=0) then
              begin
                IncludeToEligible;
                no:=Y; nod:=Node^.ind;
              end
            else if (nod>Node^.ind) then
              begin
                LocateNode(NodeTreeRoot,no);
                ExcludeFromEligible;
                no:=Y; nod:=Node^.ind;
                LocateNode(NodeTreeRoot,no);
                IncludeToEligible;
              end;
          end;

        begin    (* DetermineEligibleNode *)
          for Y:=1 TO imax DO
          begin
            LocateNode(NodeTreeRoot,Y);
            if (Node^.ActiveSet) and (Node^.ia=0) and (Node^.ind>0)
              and not (Y=T) then
              begin
                SMALLEST;
                tryother:=false;
              end;
          end;
        end;

      Procedure Eligible2;
        begin
          DetermineEligibleNode;
          if (EligibleNodes=1) then
            begin
              Y:=no;
              LocateNode(NodeTreeRoot,Y);
              IncludeToLevel;
            end;
        end;

      begin      (* FindEligibleNode *)
        tryother:=true;
        for Y:=1 TO imax DO Eligible1;
        if tryother then Eligible2;
        if tryother and (EligibleNodes=0)
          and (ActiveNodes<>0) then Delete
        else if (EligibleNodes=0) and (ActiveNodes=0)
          then Terminate:=true;
      end;       (* FindEligibleNode *)

    begin        (* NodeEligible *)
      LocateNode(NodeTreeRoot,T);
      if (Node^.ActiveSet) and (ActiveNodes=1) and not (Node^.ia=0)
        then begin Delete; ScanNodeT; end
      else if ((Node^.ActiveSet) and (Node^.ia=0)) then ScanNodeT
      else FindEligibleNode;
    end;

  Procedure IncludeArc;
    Procedure IncludeForwardArc;
      begin
        if not Arc^.EplusSet then IncludeToEplusSet;
        Update;
        yy:=Arc^.HeadNode;
        LocateNode(NodeTreeRoot,yy);
        if not (Node^.ActiveSet) then IncludeToActive;
      end;

    Procedure IncludeReverseArc;
      begin
        if not Arc^.EminusSet then IncludeToEminusSet;
        Update;
        yy:=Arc^.TailNode;
        LocateNode(NodeTreeRoot,yy);
        if not (Node^.ActiveSet) then IncludeToActive;
      end;

    Procedure Continue;
      begin
        for Y:=1 TO imax DO
        begin
          LocateNode(NodeTreeRoot,Y);
          if Node^.EligibleSet then
            begin
              for Z:=1 TO jmax DO
              begin
                LocateArc(ArcTreeRoot,Z);
                if (Y=Arc^.TailNode) then
                begin
                  if not (Arc^.DarcsSet) and
                     not (Arc^.EplusSet or Arc^.EminusSet) then
                    if (Arc^.Admissible=1) or (Arc^.Admissible=2)
                       then IncludeForwardArc;
                end
                else if (Y=Arc^.HeadNode) then
                begin
                  if not (Arc^.DarcsSet) and
                     not (Arc^.EplusSet or Arc^.EminusSet) then
                  if (Arc^.Admissible=1) or (Arc^.Admissible=3)
                    then IncludeReverseArc;
                end
              end;
              LocateNode(NodeTreeRoot,Y);
              ExcludeFromEligible;
              IncludeToScanned;
              if (EligibleNodes=0) and not (Level[C]=0) then C:=C+1;
            end;
        end;
      end;

    Procedure Omit;
      begin
        Done:=true;
        for Z:=1 TO jmax DO
        begin
          LocateArc(ArcTreeRoot,Z);
          I:=Arc^.TailNode;
          LocateNode(NodeTreeRoot,I);
          if (Node^.ActiveSet) then if Arc^.EminusSet then
            ExcludeEminusSet;
          I:=Arc^.HeadNode;
          LocateNode(NodeTreeRoot,I);
          if (Node^.ActiveSet) then if Arc^.EplusSet then
            ExcludeEplusSet;
        end;
        for I:=1 TO imax DO
        begin
          LocateNode(NodeTreeRoot,I);
          if not (Node^.ScannedSet) then IncludeToScanned;
        end;
      end;

    begin    (* IncludeArc *)
      Done:=false;
      LocateNode(NodeTreeRoot,T);
      if not (Node^.ScannedSet) then Continue
      else Omit;
    end;

  begin                    (* GenerateNetwork *)
    Terminate:=false;
    Intermediate:=true;
    repeat
      NodeEligible;
      IncludeArc;
    until Done or Terminate;
    if Terminate then Intermediate:=false;
  end;

  Procedure MaximalFlow;  (* NBF 1.102-ALGORITHM for maximal flow *)
    var MAXI,OK_FIRST:boolean;
        PassReverse,PassForward:boolean;
        Surplus:integer;

    Procedure SwitchTailHead;
      var A:integer;
      begin
        A:=Arc^.TailNode;
        Arc^.TailNode:=Arc^.HeadNode;
        Arc^.HeadNode:=A;
      end;

    Procedure IncludeToDelNodes;
      begin
         Node^.DelNodesSet:=true;
         DelNodes:=DelNodes+1;
      end;

    Procedure IncludeToDnodes;
      begin
        Node^.DnodesSet:=true; Node^.DelNodesSet:=false;
        DNodes:=DNodes+1; DelNodes:=DelNodes-1;
      end;

    Procedure FindNodePotentials;
      begin
        LocateNode(NodeTreeRoot,I);
        if (I=so) then Node^.Rho:=Node^.Beta
        else if (I=T) then Node^.Rho:=Node^.Alpha
        else if (Node^.Alpha<Node^.Beta) then Node^.Rho:=Node^.Alpha
        else Node^.Rho:=Node^.Beta;
        if (Node^.Rho=0) and not Node^.DelNodesSet then 
          IncludeToDelNodes;
      end;

    Procedure Blocking;
      begin
        if not (I=so) and not (I=T) then
        begin
          if (Node^.FlowIn>Node^.FlowOut) then Node^.BlockingStatus:=1
          else if (Node^.FlowIn<Node^.FlowOut) then Node^.BlockingStatus:=2
          else if (Node^.FlowIn=Node^.FlowOut) then Node^.BlockingStatus:=3;
        end;
      end;

    Procedure PruneTheNetwork;
      var SourcePotential,SinkPotential:integer;
      Procedure Prune1;
        begin
          if (Node^.Alpha<=Node^.Beta) and not (Node^.NodeNumber=T)
            then Node^.Rho:=Node^.Alpha
          else if (Node^.NodeNumber=T) then Node^.Rho:=Node^.Alpha;
          if (Node^.Rho=0) and not Node^.DelNodesSet
            and not Node^.DnodesSet then IncludeToDelNodes;
        end;

      Procedure Prune2;
        begin
          if (Node^.Beta<=Node^.Alpha) and not (Node^.NodeNumber=so)
            then Node^.Rho:=Node^.Beta
          else if (Node^.NodeNumber=so) then Node^.Rho:=Node^.Beta;
          if (Node^.Rho=0) and not Node^.DelNodesSet
            and not Node^.DnodesSet then IncludeToDelNodes;
        end;

      Procedure PruneFirst;
        begin
          J:=Arc^.HeadNode;
          LocateNode(NodeTreeRoot,J);
          Node^.Alpha:=Node^.Alpha-Arc^.Kappa;
          Prune1;
          IncludeToDarcs;
          Node^.FlowIn:=Node^.FlowIn-Arc^.ArcFlow;
          LocateNode(NodeTreeRoot,I);
          Node^.Beta:=Node^.Beta-Arc^.Kappa;
          Prune2;
          Node^.FlowOut:=Node^.FlowOut-Arc^.ArcFlow;
        end;

      Procedure PruneSecondly;
        begin
          J:=Arc^.TailNode;
          LocateNode(NodeTreeRoot,J);
          Node^.Beta:=Node^.Beta-Arc^.Kappa;
          Prune2;
          IncludeToDarcs;
          Node^.FlowOut:=Node^.FlowOut-Arc^.ArcFlow;
          LocateNode(NodeTreeRoot,I);
          Node^.Alpha:=Node^.Alpha-Arc^.Kappa;
          Prune1;
          Node^.FlowIn:=Node^.FlowIn-Arc^.ArcFlow;
        end;

      begin  (* PruneTheNetwork *)
        I:=1;
        repeat
          repeat
            LocateNode(NodeTreeRoot,I);
            if (Node^.DelNodesSet) and (Node^.Beta>0) then
            begin
              if not Node^.DnodesSet then IncludeToDnodes;
              for Z:=1 TO jmax DO
              begin
                  LocateArc(ArcTreeRoot,Z);
                  if (Arc^.TailNode=I) then
                    begin
                      LocateNode(NodeTreeRoot,I);
                      if (Arc^.EplusSet or Arc^.EminusSet) and not
                         (Arc^.DarcsSet) then PruneFirst;
                    end
                  else if (Arc^.HeadNode=I) and not (Arc^.DarcsSet) then
                    begin
                      LocateNode(NodeTreeRoot,I);
                      J:=Arc^.TailNode;
                      IncludeToDarcs;
                      Node^.FlowIn:=Node^.FlowIn-Arc^.ArcFlow;
                      LocateNode(NodeTreeRoot,J);
                      Node^.FlowOut:=Node^.FlowOut-Arc^.ArcFlow;
                    end;
                end;
              end
            else if (Node^.DelNodesSet) and (Node^.Alpha>0) then
              begin
                if not Node^.DnodesSet then IncludeToDnodes;
                for Z:=1 TO jmax DO
                begin
                  LocateArc(ArcTreeRoot,Z);
                  if (Arc^.HeadNode=I) then
                    begin
                      LocateNode(NodeTreeRoot,I);
                      if (Arc^.EplusSet or Arc^.EminusSet) and not
                         (Arc^.DarcsSet) then PruneSecondly;
                    end
                  else if (Arc^.TailNode=I) and not (Arc^.DarcsSet) then
                    begin
                      LocateNode(NodeTreeRoot,I);
                      J:=Arc^.HeadNode;
                      IncludeToDarcs;
                      Node^.FlowOut:=Node^.FlowOut-Arc^.ArcFlow;
                      LocateNode(NodeTreeRoot,J);
                      Node^.FlowIn:=Node^.FlowIn-Arc^.ArcFlow;
                    end;
                end;
              end
            else if (Node^.DelNodesSet) then
              begin
                if not Node^.DnodesSet then IncludeToDnodes;
                for Z:=1 TO jmax DO
                begin
                  LocateArc(ArcTreeRoot,Z);
                  if (Arc^.TailNode=I) or (Arc^.HeadNode=I) then
                    if not (Arc^.DarcsSet) then IncludeToDarcs;
                end;
              end;
            LocateNode(NodeTreeRoot,so);
            SourcePotential:=Node^.Rho;
            LocateNode(NodeTreeRoot,T);
            SinkPotential:=Node^.Rho;
            if (SourcePotential=0) or (SinkPotential=0) then MAXI:=true;
            I:=I+1;
          until MAXI or (I=imax+1) or (DelNodes=0);
          if (DelNodes<>0) then I:=1;
        until MAXI or (DelNodes=0);
      end;

    Procedure AssignFlow;
      Procedure AssignFirst;
        Procedure AssignForward;
          begin
            Arc^.Kappa:=Arc^.ArcCapacity-Arc^.ArcFlow;
            Arc^.ArcFlow:=Arc^.Kappa;
            Arc^.ArcCapacity:=Arc^.Kappa;
          end;
        Procedure AssignReverse;
          begin
            Arc^.Kappa:=Arc^.ArcFlow;
            Arc^.ArcFlow:=Arc^.Kappa;
            Arc^.ArcCapacity:=Arc^.Kappa;
            SwitchTailHead;
          end;
        begin
          for J:=1 TO jmax DO
          begin
            LocateArc(ArcTreeRoot,J);
            Arc^.TemporaryFlow:=Arc^.ArcFlow;
            Arc^.TemporaryCapacity:=Arc^.ArcCapacity;
            if (Arc^.EplusSet) then AssignForward
            else if (Arc^.EminusSet) then AssignReverse
            else Arc^.ArcFlow:=0;
          end;
        end;

      Procedure AssignSecondly;
        begin
          for J:=1 TO jmax DO
          begin
            LocateArc(ArcTreeRoot,J);
            I:=Arc^.TailNode;
            LocateNode(NodeTreeRoot,I);
            Node^.FlowOut:=Node^.FlowOut+Arc^.ArcFlow;
            Node^.Beta:=Node^.FlowOut;
            I:=Arc^.HeadNode;
            LocateNode(NodeTreeRoot,I);
            Node^.FlowIn:=Node^.FlowIn+Arc^.ArcFlow;
            Node^.Alpha:=Node^.FlowIn;
          end;
          for I:=1 TO imax DO FindNodePotentials;
        end;

      begin  (* AssignFlow *)
        AssignFirst; AssignSecondly;
        PruneTheNetwork;
        for J:=1 TO jmax DO
          begin
            LocateArc(ArcTreeRoot,J);
            if not Arc^.DarcsSet then Arc^.Kappa:=0
            else if Arc^.DarcsSet then Arc^.ArcFlow:=0;
          end;
        for I:=1 TO imax DO
        begin
          LocateNode(NodeTreeRoot,I);
          Node^.Alpha:=0; Node^.Beta:=0; Node^.Rho:=0;
          if not (Node^.DnodesSet) then Blocking;
        end;
      end;

    Procedure MakeNewAssignment;
      begin
        for J:=1 TO jmax DO
        begin
          LocateArc(ArcTreeRoot,J);
          if not (Arc^.DarcsSet) then
            begin
              if (Arc^.EplusSet) then
                begin
                  Arc^.ArcFlow:=Arc^.TemporaryFlow+Arc^.ArcFlow;
                  Arc^.TemporaryFlow:=Arc^.ArcFlow;
                  Arc^.ArcFlow:=Arc^.Kappa;
                  Arc^.Kappa:=0;
                end
              else if (Arc^.EminusSet) then
                begin
                  Arc^.ArcFlow:=Arc^.TemporaryFlow-Arc^.ArcFlow;
                  Arc^.TemporaryFlow:=Arc^.ArcFlow;
                  Arc^.ArcFlow:=Arc^.Kappa;
                  Arc^.Kappa:=0;
                end
              else
                begin
                  Arc^.TemporaryFlow:=Arc^.ArcFlow;
                  Arc^.ArcFlow:=0;
                  Arc^.Kappa:=0;
                end;
            end;
        end;
        for I:=1 TO imax DO
        begin
          LocateNode(NodeTreeRoot,I);
          if not (Node^.DnodesSet) then
            begin
              Node^.FlowIn:=Node^.Alpha;
              Node^.FlowOut:=Node^.Beta;
              Blocking;
              Node^.Alpha:=0; Node^.Beta:=0; Node^.Rho:=0;
            end;
        end;
      end;

    Procedure ChangeBack;
      begin
        for J:=1 TO jmax DO
        begin
          LocateArc(ArcTreeRoot,J);
          Arc^.ArcCapacity:=Arc^.TemporaryCapacity;
          if (Arc^.EplusSet) then
            Arc^.ArcFlow:=Arc^.TemporaryFlow+Arc^.ArcFlow
          else if (Arc^.EminusSet) then
            begin
              Arc^.ArcFlow:=Arc^.TemporaryFlow-Arc^.ArcFlow;
              SwitchTailHead;
            end
          else Arc^.ArcFlow:=Arc^.TemporaryFlow;
        end;
      end;
   
    Procedure Surp1;
      begin
        Node^.FlowIn:=Node^.FlowIn-Surplus;
        Node^.Alpha:=Node^.Alpha+Surplus;
      end;

    Procedure Surp2;
      begin
        Node^.FlowOut:=Node^.FlowOut-Surplus;
        Node^.Beta:=Node^.Beta+Surplus;
      end;

    Procedure ChangeSurplus1;
      begin
        Surp1;
        LocateNode(NodeTreeRoot,G);
        Surp2;
      end;

    Procedure ChangeSurplus2;
      begin
        Surp2;
        LocateNode(NodeTreeRoot,G);
        Surp1;
      end;

    Procedure ChangeFlow1;
      begin
        Arc^.ArcFlow:=Arc^.ArcFlow-Excess;
        Arc^.Kappa:=Arc^.Kappa+Excess;
        Surplus:=Excess;
        if PassForward then ChangeSurplus1
        else if PassReverse then ChangeSurplus2;
        LocateNode(NodeTreeRoot,G);
        Node^.BlockingStatus:=3;
        LocateNode(NodeTreeRoot,I);
        Blocking;
      end;

    Procedure ChangeFlow2;
      begin
        Arc^.Kappa:=Arc^.Kappa+f1;
        Arc^.ArcFlow:=0;
        Excess:=Excess-f1;
        Surplus:=f1;
        if PassForward then ChangeSurplus1
        else if PassReverse then ChangeSurplus2;
        LocateNode(NodeTreeRoot,I);
        Blocking;
      end;

    Procedure FlowChange;
      begin
        f1:=Arc^.ArcFlow;
        if (f1>=Excess) then ChangeFlow1
        else if (f1<Excess) and (f1>0) then ChangeFlow2;
      end;

    Procedure NoForwardPass;
      begin
        for J:=1 TO jmax DO
        begin
          LocateArc(ArcTreeRoot,J);
          if (Arc^.TailNode=so) and (Arc^.HeadNode=T) then
            Arc^.ArcFlow:=Arc^.ArcCapacity
          else Arc^.ArcFlow:=0;
          OK_FIRST:=true;
        end;
      end;

    Procedure DoBackwardPass;
      var ReduceToBFNodes,ReduceToPFandBALNodes,ReduceFromSource:boolean;
      Procedure BackwardReduce;
        begin
          J:=1;
          begin
            repeat
              LocateArc(ArcTreeRoot,J);
              if not (Arc^.DarcsSet) and (Arc^.HeadNode=G) then
              begin
                I:=Arc^.TailNode;
                LocateNode(NodeTreeRoot,I);
                if not (Node^.DnodesSet) then
                begin
                  if ReduceToBFNodes and not (I=so) and (Node^.BlockingStatus=2)
                    then FlowChange
                  else if ReduceToPFandBALNodes and not (I=so) then FlowChange
                  else if ReduceFromSource and (I=so) then FlowChange;
                end;
              end;
              LocateNode(NodeTreeRoot,G);
              J:=J+1;
            until (J=jmax+1) or (Node^.BlockingStatus=3);
          end;
        end;

      Procedure BackPass;
          Procedure B_pass2;
            begin
              Excess:=Node^.FlowIn-Node^.FlowOut;
              if (Excess>0) then
                begin
                  BackwardReduce;
                  ReduceToBFNodes:=false; ReduceToPFandBALNodes:=true;
                  if not (Node^.BlockingStatus=3) then BackwardReduce;
                  ReduceToPFandBALNodes:=false; ReduceFromSource:=true;
                  if not (Node^.BlockingStatus=3) then BackwardReduce;
                  ReduceToBFNodes:=true; ReduceFromSource:=false;
                end;
              C:=C-1;
            end;


        begin  (*  BackPass  *)
          repeat
            LocateNode(NodeTreeRoot,G);
            if (Node^.BlockingStatus=3) or
               (Node^.BlockingStatus=2) or
               (Node^.BlockingStatus=0) then C:=C-1
            else B_pass2;
            G:=Level[C];
          until (C=0);
        end;

      begin  (* DoBackwardPass *)
        PassReverse:=true;
        ReduceToBFNodes:=true;
        ReduceToPFandBALNodes:=false;
        ReduceFromSource:=false;
        C:=cmax-1;
        G:=Level[C];
        if (cmax=1) then NoForwardPass
        else BackPass;
        PassReverse:=false;
      end;   (* DoBackwardPass *)

    Procedure DoForwardPass;
      var ReduceToBFandBALNodes, ReduceToSink:boolean;
      Procedure ForwardReduce;
        begin
          J:=1;
          begin
            repeat
              LocateArc(ArcTreeRoot,J);
              if not (Arc^.DarcsSet) and (Arc^.TailNode=G) then
                begin
                  I:=Arc^.HeadNode;
                  LocateNode(NodeTreeRoot,I);
                  if not (Node^.DnodesSet) then
                  begin
                    if ReduceToBFandBALNodes and not (I=T) then FlowChange
                    else if ReduceToSink and (I=T) then FlowChange;
                  end;
                end;
              LocateNode(NodeTreeRoot,G);
              J:=J+1;
            until (J=jmax+1) or (Node^.BlockingStatus=3);
          end;
          ReduceToBFandBALNodes:=false; ReduceToSink:=true;
        end;

      Procedure ForPass;
      begin
        repeat
          begin 
            LocateNode(NodeTreeRoot,G);
            if (Node^.BlockingStatus=3) or
               (Node^.BlockingStatus=1) or
               (Node^.BlockingStatus=0) then C:=C+1
            else
            begin
              Excess:=Node^.FlowOut-Node^.FlowIn;
              if (Excess>0) then
              begin
                ForwardReduce;
                if not (Node^.BlockingStatus=3) then ForwardReduce;
                ReduceToBFandBALNodes:=true; ReduceToSink:=false;
              end;
            end;
          end;
          G:=Level[C];
        until (C=cmax);
      end;

      begin  (* forward pass *)
        PassForward:=true;
        ReduceToBFandBALNodes:=true; ReduceToSink:=false;
        C:=1;
        G:=Level[C];
        ForPass;
        PassForward:=false;
      end;   (* forward pass *)

    begin    (* NBF 1.1-Algorithm for MaximalFlow *)
      PassReverse:=false; PassForward:=false; MAXI:=false; OK_FIRST:=false;
      nr:=0;
      AssignFlow;
      if not MAXI then
        begin
          repeat
            DoBackwardPass;
            if not OK_FIRST or MAXI then DoForwardPass
            else if OK_FIRST then MAXI:=true;
            nr:=nr+1;
            for I:=1 TO imax DO
            begin
              LocateNode(NodeTreeRoot,I);
              if not (Node^.DnodesSet) then FindNodePotentials;
            end;
            PruneTheNetwork;
            if not MAXI then MakeNewAssignment;
          until MAXI;
        end;
      ChangeBack;
    end;

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

Procedure WriteOut;
  begin
    writeln(J:6,Arc^.TailNode:6,Arc^.HeadNode:6,
                Arc^.ArcFlow:6,Arc^.ArcCapacity:6);
    if WriteResults then
      writeln(Netres,J:6,Arc^.TailNode:6,Arc^.HeadNode:6,
                         Arc^.ArcFlow:6,Arc^.ArcCapacity:6)
    else
    begin Line:=Line+1; if (Line=10) then WayToGo;
    end;
  end;

Procedure Results1;
  begin
    Banner;
    writeln;
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
      writeln('Flow values on arcs');
      writeln('    Arc   From  To   Flow   Cap');
      writeln;
      if WriteResults then
        begin
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
        end;
      writeln;
      writeln('Note: The flow is zero on all arcs not ');
      writeln('listed above.');
      GoOn;
    end;

  Procedure ArcsEplus;
    var FFP:integer;
    begin
      Line:=0;
      writeln;
      writeln('The set E+ contains the following arcs:');
      writeln('    Arc   From  To   Flow   Kappa');
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
              if (Line=15) then WayToGo;
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
              if (Line=15) then WayToGo;
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
          if (so=Arc^.TailNode) then MaxFlow:=MaxFlow+Arc^.ArcFlow
          else if (so=Arc^.HeadNode) then MaxFlow:=MaxFlow-Arc^.ArcFlow;
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

  Procedure FindCutset;  
    var POS,II:integer;
    begin
      Line:=0;
      POS:=0;
      ClrScr;
      writeln('The set X of the partition  X,X-bar ');
      writeln('contains following nodes');
      writeln('{');
      for I:=1 TO imax DO
      begin
        LocateNode(NodeTreeRoot,I);
        if (Node^.ScannedSet) then
        begin
          if POS<11 then
            begin
              write(I:5); POS:=POS+1;
            end
          else
            begin
              writeln; POS:=0; write(I:5);
            end;
        end;
      end;
      writeln; writeln('}'); writeln;
      writeln('The cutset contains following arcs');
      writeln('    Arc   From  To   Flow   Cap');
      if WriteResults then
        begin
          writeln(Netres);
          writeln(Netres,'The set X of the partition ');
          writeln(Netres,'X,X-bar contains following nodes');
begin
  POS:=0;
  writeln(Netres,'{');
  for I:=1 TO imax DO
  begin
    LocateNode(NodeTreeRoot,I);
    if (Node^.ScannedSet) then
    begin
      if POS<11 then
        begin
          write(Netres,I:5); POS:=POS+1;
        end
      else
        begin
          writeln(Netres); POS:=0; write(Netres,I:5);
        end;
    end;
  end;
  writeln(Netres);
  writeln(Netres,'}');
  writeln(Netres);
  writeln(Netres,'    The cutset contains following arcs');
  writeln(Netres,'    Arc   From  To   Flow   Cap');
end;
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
      GoOn;
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

  Procedure EmptyLines(F:integer);
    begin
      I:=1;
      repeat writeln; I:=I+1;
      until I=F;
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
          write('Enter input file name: ');
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
        write('Enter name of output file: ');
        readln(OutFileName);
        ASSIGN(Netres,OutFileName);
        REwrite(Netres);
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

Procedure writeMenu;
begin
  ClrScr;
  writeln('------------------------------------------');
  writeln('        Network Generator 1.0');
  writeln('        Max-Flow Algorithm 2.4');
  writeln('           Uses MIN(AD(i))          ');
  writeln('    USES TREES AS DATA STRUCTURES');
  writeln('           ***  MENU  ***           ');
  writeln('------------------------------------------');
  writeln('  1) Explain this program');
  Delay(250);
  writeln('  2) Show example case');
  Delay(250);
  writeln('  3) Display input');
  Delay(250);
  writeln('  4) Retrieve data from a file');
  Delay(250);
  writeln('  5) Solve');
  Delay(250);
  writeln('  6) Save final results to a file');
  Delay(250);
  writeln('  7) Return to DOS');
end;

Procedure InitPage(PAGE,LASTPAGE:integer);
begin
  writeln('PROGRAM EXPLANATION     PAGE ',PAGE,' OF ',LASTPAGE);
  writeln('PRESS:    C             B          E   ');
  writeln('       CONTINUE      BACK UP      EXIT ');
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
writeln('structures (balanced trees) to store');
writeln('and update network data: a node tree and');
writeln('an arc tree. Data can be read from (and');
writeln('written to) a user file. Be sure, when ');
writeln('using your own files, that the data file');
writeln('does not contain any empty lines, and ');
writeln('that the nodes and arcs are numbered ');
writeln('consecutively with no number missing ');
writeln('in-between. Numbering of nodes and arcs');
writeln('may be done in any order, but source node');
writeln('must be assigned number 1, and sink node');
writeln('the largest node number. Initial flow on');
writeln('all arcs is assumed to be zero.');
  end;
  Procedure PAGE3;
  begin
writeln('FORMAT OF A NETWORK DATA FILE:');
writeln('Line c:  Comment line');
writeln('Line p:  Problem line. Gives the type of');
writeln('         the problem (max), as well as');
writeln('         the number of nodes and arcs.');
writeln('Line n:  Node line. Gives the source node,');
writeln('         and a node designator s.');
writeln('Line n:  Node line. Gives the sink node,');
writeln('         and a node designator t.');
writeln('Line a:  Arc line. Several arc lines. Each');
writeln('         line consists of three numbers:');
writeln('         Arc Tail-Arc-Head-Flow Capacity');
writeln('Before attempting your own data, try to');
writeln('create a data file and read it in. Use the');
writeln('data given on the next page. Your results');
writeln('should correspond to those of the example');
writeln('case of menu item 2.');
  end;
  Procedure PAGE4;
  begin
writeln('c EXAMPLE NETWORK DATA FILE');
writeln('p max  7     12 ');
writeln('n      1      s ');
writeln('n      7      t ');
writeln('a      1      2      5');
writeln('a      1      3     13');
writeln('a      1      4      8');
writeln('a      3      2      2');
writeln('a      3      4      7');
writeln('a      2      5      6');
writeln('a      3      5      1');
writeln('a      4      5      4');
writeln('a      2      6     10');
writeln('a      5      7      9');
writeln('a      4      7      6');
writeln('a      6      7      6');
  end;

begin
  PG := 1;
  repeat
    ClrScr;
    InitPage(PG,4);
    CASE PG OF
      1: PAGE1; 2: PAGE2; 3: PAGE3; 4: PAGE4;
    end;
    GotoXY(4,3);
    readln(ch);
    if (ch = 'C') or (ch = 'c') then PG := PG + 1
    else if (ch = 'B') or (ch = 'b') then PG := PG - 1
  until (ch = 'E') or (ch = 'e') or (PG=5) or (PG=0)
end;

  begin
    xample:=false;
    SaveData:=false;
    Values:=false; Intermediate:=false;
    IntermediateResults:=false;
    WriteResults:=false;
    InitMax;
    repeat
      writeMenu;
      writeln; writeln;
      write('Select by number and press RETURN ');
      readln(ch);
      writeln;
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
  end.
