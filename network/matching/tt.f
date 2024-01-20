      PROGRAM TT
C     GENERATES A CHAIN OF TRIANGLES CONNECTED AT ALL VERTICES
C     EXAMPLE:
C      1--------4-------7-------10--------13
C      º \      º \     º \     º \       º  \
C      º   3----+---6---+---9---+---12----+----15
C      º /      º /     º /     º /       º  /
C      2--------5-------8-------11--------14
C
C     DATE:         7-1-91
C     LANGUAGE:     FORTRAN 77
C     PROGRAMMERS:  N. RITCHEY
C                   B. MATTINGLY
C                   YOUNGSTOWN STATE UNIVERSITY
C
C     INPUT:  K = NUMBER OF TRIANGLES
C
C     OUTPUT:  NUMBER OF VERTICES = N = 3*K
C              NUMBER OF EDGES    = M = 6*K - 3
C              LIST OF EDGES   (EDGE COSTS ARE SET = 1)
C
      READ(5, *) K
      N = 3 * K
      M = 6 * K - 3
C
      WRITE(6,11)N,M
11    FORMAT('p edge ', 2I10)
      L = 1
5     I1 = 3*(L-1) + 1
      I2 = I1 + 1
C
      DO 20 I = I1,I2
         WRITE(6,21)I,I+1
   21    FORMAT('e ',2I10,'     1')
         IF(I.LT.I2) WRITE(6,21)I,I+2
20    CONTINUE
C
      IF(L.LT.K)THEN
        L = L + 1
        GO TO 5
      ENDIF
      DO 25 I = 1,N-3
         WRITE(6,21)I,I+3
25    CONTINUE
      STOP
      END
