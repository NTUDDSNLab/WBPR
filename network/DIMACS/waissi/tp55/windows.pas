UNIT Windows;

{$D-,S-}

INTERFACE

Uses Crt;

type

  TitleStr = string[63];
  FrameChars = array[1..8] of Char;
  WinState = record
    WindMin, WindMax: Word;
    WhereX, WhereY: Byte;
    TextAttr: Byte;
  end;

const

  SingleFrame: FrameChars = 'ÚÄ¿³³ÀÄÙ';
  DoubleFrame: FrameChars = 'ÉÍ»ººÈÍ¼';

procedure WriteStr(X, Y: Byte; S: String; Attr: Byte);
procedure WriteChar(X, Y, Count: Byte; Ch: Char; Attr: Byte);

procedure FillWin(Ch: Char; Attr: Byte);
procedure ReadWin(var Buf);
procedure WriteWin(var Buf);
function WinSize: Word;
procedure SaveWin(var W: WinState);
procedure RestoreWin(var W: WinState);
procedure FrameWin(Title: TitleStr; var Frame: FrameChars;
  TitleAttr, FrameAttr: Byte);
procedure UnFrameWin;

IMPLEMENTATION

{$L WINDOWS}

procedure WriteStr(X, Y: Byte; S: String; Attr: Byte);
external {WINDOWS};

procedure WriteChar(X, Y, Count: Byte; Ch: Char; Attr: Byte);
external {WINDOWS};

procedure FillWin(Ch: Char; Attr: Byte);
external {WINDOWS};

procedure WriteWin(var Buf);
external {WINDOWS};

procedure ReadWin(var Buf);
external {WINDOWS};

function WinSize: Word;
external {WINDOWS};

procedure SaveWin(var W: WinState);
begin
  W.WindMin := WindMin;
  W.WindMax := WindMax;
  W.WhereX := WhereX;
  W.WhereY := WhereY;
  W.TextAttr := TextAttr;
end;

procedure RestoreWin(var W: WinState);
begin
  WindMin := W.WindMin;
  WindMax := W.WindMax;
  GotoXY(W.WhereX, W.WhereY);
  TextAttr := W.TextAttr;
end;

procedure FrameWin(Title: TitleStr; var Frame: FrameChars;
  TitleAttr, FrameAttr: Byte);
var
  W, H, Y: Word;
begin
  W := Lo(WindMax) - Lo(WindMin) + 1;
  H := Hi(WindMax) - Hi(WindMin) + 1;
  WriteChar(1, 1, 1, Frame[1], FrameAttr);
  WriteChar(2, 1, W - 2, Frame[2], FrameAttr);
  WriteChar(W, 1, 1, Frame[3], FrameAttr);
  if Length(Title) > W - 2 then Title[0] := Chr(W - 2);
  WriteStr((W - Length(Title)) shr 1 + 1, 1, Title, TitleAttr);
  for Y := 2 to H - 1 do
  begin
    WriteChar(1, Y, 1, Frame[4], FrameAttr);
    WriteChar(W, Y, 1, Frame[5], FrameAttr);
  end;
  WriteChar(1, H, 1, Frame[6], FrameAttr);
  WriteChar(2, H, W - 2, Frame[7], FrameAttr);
  WriteChar(W, H, 1, Frame[8], FrameAttr);
  Inc(WindMin, $0101);
  Dec(WindMax, $0101);
end;

procedure UnFrameWin;
begin
  Dec(WindMin, $0101);
  Inc(WindMax, $0101);
end;

end.
