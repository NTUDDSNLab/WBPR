{}
{   Gary R. Waissi, Copyright (C) (1990,1991)      }
{}
{ This unit contains the variable declarations for }
{ the NBF - Max-Flow Algorithm in MxiSub1, and     }
{ for the network generator in MxiSub2             }
{ This unit also contains the tree building, tree  }
{ search, and tree print procedures for both the   }
{ arc tree and the node tree.                      }
{}
UNIT MxiSub4;

INTERFACE
  uses crt;

  const
    nmax = 30000;
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
    ch                 :char;          

    Level              :array[0..nmax] of integer;

    OutFileName        :string[10];
    Netres             :text;
    InFileName         :string[20];
    InputData          :text;
    problem            :string[5];

FUNCTION NodeTree(n:integer):NodeTreeType;
FUNCTION ArcTree(nn:integer):ArcTreeType;
PROCEDURE LocateNode(t1:NodeTreeType; h1:INTEGER);
PROCEDURE LocateArc(t2:ArcTreeType; h2:INTEGER);
PROCEDURE PrintNodeTree(NodeTree1:NodeTreeType);
PROCEDURE PrintArcTree(ArcTree1:ArcTreeType);
PROCEDURE WayToGo;
PROCEDURE GoOn;
PROCEDURE NetworkLevels;

IMPLEMENTATION

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
      if (Line=10) then
      begin
        WayToGo;
        ClrScr;
        writeln;
        writeln('Levels of the Acyclic Network');
        writeln;
      end;
      C:=C+1;
    until (C=cmax);
    GoOn;
  end;

end.
