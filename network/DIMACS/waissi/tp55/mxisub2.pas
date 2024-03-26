{}
{    Gary R. Waissi, Copyright (C) (1990,1991)       }
{}
{ This Unit contains the Acyclic Network Generator.  }
{ An acyclic network is generated from an original   }
{ network in the Phase I. The max-flow algorithm is  }
{ then in the Phase II applied to this network (see  }
{ Unit Flow 2 for the Max-Flow Algorithm).           }
{ The Phases I and II are repeated until the maximum }
{ value flow is found in the original network.       }
{}

UNIT MxiSub2;

INTERFACE
  
  uses DOS,CRT,MxiSub3,MxiSub4;
  Procedure Initialize;
  Procedure Prepare;
  Procedure GenerateNetwork;

IMPLEMENTATION

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

end.
