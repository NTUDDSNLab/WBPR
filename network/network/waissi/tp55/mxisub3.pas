{}
{      Gary R. Waissi, Copyright (C) (1990,1991)          }
{}
{ This Unit contains small routines to update the status  }
{ of nodes and arcs while network generation and max-flow }
{ algorithms are prcessing the network.                   }
{}

UNIT MxiSub3;

INTERFACE
  uses crt,mxisub4;

Procedure IncludeToActive;
Procedure ExcludeFromActive;
Procedure IncludeToEligible;
Procedure ExcludeFromEligible;
Procedure IncludeToScanned;
Procedure IncludeToDarcs;
Procedure IncludeToEplusSet;
Procedure ExcludeEplusSet;
Procedure IncludeToEminusSet;
Procedure ExcludeEminusSet;

IMPLEMENTATION

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

end.
