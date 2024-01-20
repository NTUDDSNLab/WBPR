{
 Gary R. Waissi, University of Michigan-Dearborn,
 School of Management, FOB 113, Dearborn, MI 48128.
 e-mail: gary_waissi@um.cc.umich.edu
 tel   : (313) 593-5012

 This is the MAIN PROGRAM. The program calls the five
 network generators and the max-flow solver from files:

 AcMaxi.pas       Acyclic network generator
 BiMaxi.pas       Bipartite network generator
 Tr1Maxi.pas      Transit grid (one-way) generator
 Tr2Maxi.pas      Transit grid (two-way) generator
 RaMaxi.pas       Random network generator.

 MxiMain.pas      Main program of the max-flow solver.
                  This program calls the follwing
                  subprograms:

    MxiSub1.pas   Max-flow algorithm
    MxiSub2.pas   Network generator
    MxiSub3.pas   Common small procedures
    MxiSub4.pas   Variable declarations, tree build,
                  tree search, tree print procedures.

 Compilation of the program 

 The program can be compiled "as is" using the TURBO
 Pascal 5.5. Please use the following steps:

 1. Select - Options/Compiler/Overlays allowed   ON
 2. Select - Options/Linker/Link buffer          DISK
 3. Select - Compiles/Destination                DISK
 4. Compile using - Compiler/Build

 The first four network generators are compiled in
 OVERLAY mode to save memory.

}
{$F+}
program MXNETGEN(Input,Output);
{$M 20000,0,512000}  

{$S-}

uses Overlay, Crt, Windows, Dos,
     AcMaxi,  BiMaxi,  Tr1Maxi, Tr2Maxi, RaMaxi,
     MxiMain;
     {$O AcMaxi}
     {$O BiMaxi}
     {$O Tr1Maxi}
     {$O Tr2Maxi}

const

  CAcyclic   = ^A;
  CBipartite = ^B;
  CTransit1  = ^T;
  CTransit2  = ^Y;
  CRandomNet = ^R;
  CMaxFlow   = ^M;
  CClose     = ^C;
  CRight     = ^D;
  CUp        = ^E;
  CEnter     = ^M;
  CInsLin    = ^N;
  COpen      = ^O;
  CLeft      = ^S;
  CDown      = ^X;
  CDelLin    = ^Y;
  CExit      = ^[;

type

  TitleStrPtr = ^TitleStr;

  WinRecPtr = ^WinRec;
  WinRec = record
    Next: WinRecPtr;
    State: WinState;
    Title: TitleStrPtr;
    TitleAttr, FrameAttr: Byte;
    Buffer: Pointer;
  end;

var
  TopWindow: WinRecPtr;
  WindowCount: Integer;
  Done: Boolean;
  Ch: Char;
  cap:real;
  y,pp,q,i,k,num,try,arcnumber:integer;
  node_i,node_j,num_arcs,num_nodes,capacity:integer;
  tail,head:integer;
  special,example:boolean;
  DoContinue:boolean;
  OutFileName:string[20];
  netres:text;
  source:integer;
  sink:integer;

procedure RunDos;
var
  Command: string[127];

begin
  repeat
    Writeln('You may run any program in this window.');
    Writeln('Including DOS commands.');
    Write('Enter command: ');
    ReadLn(Command);
    if Command <> '' then
    begin
      SwapVectors;
      Exec(GetEnv('COMSPEC'), '/C ' + Command);
      SwapVectors;
      if DosError <> 0 then
        WriteLn('Could not execute COMMAND.COM');
      WriteLn;
    end;
  until Command = '';
end;

procedure ActiveWindow(Active: Boolean);
begin
  if TopWindow <> nil then
  begin
    UnFrameWin;
    with TopWindow^ do
      if Active then
        FrameWin(Title^, DoubleFrame, TitleAttr, FrameAttr)
      else
        FrameWin(Title^, SingleFrame, FrameAttr, FrameAttr);
  end;
end;

procedure OpenWindow(X1, Y1, X2, Y2: Byte; T: TitleStr;
  TAttr, FAttr: Byte);
var
  W: WinRecPtr;
begin
  ActiveWindow(False);
  New(W);
  with W^ do
  begin
    Next := TopWindow;
    SaveWin(State);
    GetMem(Title, Length(T) + 1);
    Title^ := T;
    TitleAttr := TAttr;
    FrameAttr := FAttr;
    Window(X1, Y1, X2, Y2);
    GetMem(Buffer, WinSize);
    ReadWin(Buffer^);
    FrameWin(T, DoubleFrame, TAttr, FAttr);
  end;
  TopWindow := W;
  Inc(WindowCount);
end;

procedure CloseWindow;
var
  W: WinRecPtr;
begin
  if TopWindow <> nil then
  begin
    W := TopWindow;
    with W^ do
    begin
      UnFrameWin;
      WriteWin(Buffer^);
      FreeMem(Buffer, WinSize);
      FreeMem(Title, Length(Title^) + 1);
      RestoreWin(State);
      TopWindow := Next;
    end;
    Dispose(W);
    ActiveWindow(True);
    Dec(WindowCount);
  end;
end;

procedure Initialize;
begin
  CheckBreak := False;
  if (LastMode <> CO80) and (LastMode <> BW80) and
    (LastMode <> Mono) then TextMode(CO80);
  TextAttr := Black + LightGray * 16;
  Window(1, 3, 80, 25);
  FillWin(#177, LightGray + Black * 16);
  Window(1, 1, 80, 25);
  GotoXY(1, 1); Write('  Network Generation for Max-Flow Problems');
  ClrEol;
  GotoXY(50,1); Write('  by Gary R. Waissi'); ClrEol;
  GotoXY(1, 2); Write('  Copyright (C) (1990,1991)');
  ClrEol;
  GotoXY(50, 2); Write('  UM-D, School of Management');
  ClrEol;
  Window(50, 3, 80, 21);
  GotoXY(1,2); Write('  HELP:'); ClrEol;
  GotoXY(1,3); Write('  To run programs'); ClrEol;
  GotoXY(1,4); Write('  PRESS Alt and a <key>'); ClrEol;
  GotoXY(1,5); Write('  where <key>=A,B,T,Y,R,M');
  ClrEol;
  GotoXY(1,6);  ClrEol;
  GotoXY(1,7);  Write('  Esc   Return to DOS'); ClrEol;
  GotoXY(1,8);  Write('  Alt-O Open Window'); ClrEol;
  GotoXY(1,9);  Write('  Alt-C Close Windows and'); ClrEol;
  GotoXY(1,10); Write('        return to MENU'); ClrEol;
  GotoXY(1,11); ClrEol;
  GotoXY(1,12); Write('  Alt-A Acyclic'); ClrEol;
  GotoXY(1,13); Write('  Alt-B Bipartite'); ClrEol;
  GotoXY(1,14); Write('  Alt-T TransitOneWay'); ClrEol;
  GotoXY(1,15); Write('  Alt-Y TransitTwoWay'); ClrEol;
  GotoXY(1,16); Write('  Alt-R Random'); ClrEol;
  GotoXY(1,17); Write('  Alt-M Maximum Flow Solver'); ClrEol;
  GotoXY(1,18); ClrEol;
  GotoXY(1,19); ClrEol;
  GotoXY(1,20); ClrEol;
  GotoXY(1,21); ClrEol;
  Window(50,23, 80, 25);
  GotoXY(1,1); Write('ACTION: ');
  TopWindow := nil;
  WindowCount := 0;
end;

procedure CreateWindow;
var
  XX1, YY1, XX2, YY2: Integer;
  S: string[15];
  Color: Byte;
begin
  Str(WindowCount + 1, S);
  begin
    XX1 := 1; YY1 := 4+WindowCount;
    XX2 := 46; YY2 := 24;
  end;
  if LastMode <> CO80 then
    Color := Black else Color := WindowCount mod 6 + 1;
  OpenWindow(XX1, YY1, XX2, YY2,'',
    Color + LightGray * 16, LightGray + Color * 16);
  TextAttr := LightGray;
  ClrScr;
end;

function ReadChar: Char;
var
  Ch: Char;
begin
  Ch := ReadKey;
  if Ch = #0 then
    case ReadKey of
      #19: Ch := CRandomNet;{ Alt-R }
      #20: Ch := CTransit1; { Alt-T }
      #21: Ch := CTransit2; { Alt-Y }
      #24: Ch := COpen;     { Alt-O }
      #30: Ch := CAcyclic;  { Alt-A }
      #45: Ch := CExit;     { Alt-X }
      #46: Ch := CClose;    { Alt-C }
      #48: Ch := CBipartite;{ Alt-B }
      #50: Ch := CMaxFlow;  { Alt-M }
      #72: Ch := CUp;       { Up }
      #75: Ch := CLeft;     { Left }
      #77: Ch := CRight;    { Right }
      #80: Ch := CDown;     { Down }
      #82: Ch := CInsLin;   { Ins }
      #83: Ch := CDelLin;   { Del }
    end;
  ReadChar := Ch;
end;

procedure Beep;
begin
  Sound(500); Delay(25); NoSound;
end;

procedure AcMax;
begin
  GotoXY(9,1);
  write('ACYCLIC');
  ClrEol;
  CreateWindow;
  AcMaxiNet;
  CloseWindow;
  if AcMaxi.DoContinue then
  begin
    GotoXY(1,2);
    Write('FILE  : ',AcMaxi.OutFileName,'.max');
    ClrEol;
  end;
end;

procedure BiMax;
begin
  GotoXY(9,1);
  write('BIPARTITE');
  ClrEol;
  CreateWindow;
  BiMaxiNet;
  CloseWindow;
  if BiMaxi.DoContinue then
  begin
    GotoXY(1,2);
    Write('FILE  : ',BiMaxi.OutFileName,'.max');
    ClrEol;
  end;
end;

procedure Tr1Max;
begin
  GotoXY(9,1);
  write('TRANSIT ONE-WAY');
  ClrEol;
  CreateWindow;
  Tr1MaxiNet;
  CloseWindow;
  if Tr1Maxi.DoContinue then
  begin
    GotoXY(1,2);
    Write('FILE  : ',Tr1Maxi.OutFileName,'.max');
    ClrEol;
  end;
end;

procedure Tr2Max;
begin
  GotoXY(9,1);
  write('TRANSIT TWO-WAY');
  ClrEol;
  CreateWindow;
  Tr2MaxiNet;
  CloseWindow;
  if Tr2Maxi.DoContinue then
  begin
    GotoXY(1,2);
    Write('FILE  : ',Tr2Maxi.OutFileName,'.max');
    ClrEol;
  end;
end;

procedure RaMax;
begin
  GotoXY(9,1);
  write('RANDOM');
  ClrEol;
  CreateWindow;
  RaMaxiNet;
  CloseWindow;
  if RaMaxi.DoContinue then
  begin
    GotoXY(1,2);
    Write('FILE  : ',RaMaxi.OutFileName,'.max');
    ClrEol;
  end;
end;

procedure MxFlow;
begin
  GotoXY(9,1);
  write('MAX-FLOW SOLVER');
  ClrEol;
  CreateWindow;
  FindMaximalFlow;
  CloseWindow;
  {if MxiSub1.DoContinue then
  begin
    GotoXY(1,2);
    Write('FILE  : ',MxiSub1.OutFileName);
    ClrEol;
  end;
  }
end;

begin
  Initialize;
  OvrInit('mxnetgen.ovr');
  DoContinue:=false;
  Done := false;
  repeat
    Ch := ReadChar;
    case Ch of
      #32..#255: Write(Ch);
      COpen: CreateWindow;
      CClose: CloseWindow;
      CUp: GotoXY(WhereX, WhereY - 1);
      CLeft: GotoXY(WhereX - 1, WhereY);
      CRight: GotoXY(WhereX + 1, WhereY);
      CDown: GotoXY(WhereX, WhereY + 1);
      CAcyclic: AcMax;
      CBipartite: BiMax;
      CTransit1: Tr1Max;
      CTransit2: Tr2Max;
      CRandomNet: RaMax;
      CMaxFlow: MxFlow;
      CInsLin: InsLine;
      CDelLin: DelLine;
      CEnter: WriteLn;
      CExit: Done := True;
    else
      Beep;
    end;
  until Done;
  Window(1, 1, 80, 25);
  NormVideo;
  ClrScr;
end.
