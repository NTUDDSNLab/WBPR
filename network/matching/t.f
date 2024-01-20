      PROGRAM T
C     GENERATES A CHAIN OF TRIANGLES CONNECTED AT ONE VERTEX
C     EXAMPLE:
C      1--------4       7       10--------13
C      º \      º \     º \     º \       º  \
C      º   3    º   6   º   9---+---12    º    15
C      º /      º /     º /     º /       º  /
C      2        5-------8       11        14
C
C     DATE:         7-1-91
C     LANGUAGE:     FORTRAN 77
C     PROGRAMMERS:  N. RITCHEY
C                   B. MATTINGLY
C                   YOUNGSTOWN STATE UNIVERSITY
C
C     INPUT:  K = NUMBER OF TRIANGLES
C
C      OUTPUT:  NUMBER OF VERTICES =N= 3*K
C               NUMBER OF EDGES    =M= 4*K-1
C               LIST OF EDGES  (EDGE COSTS ARE SET = 1)
C
      READ(5,*) K
      N = 3*K
      M = 4*K - 1
C
      WRITE(6,11) N,M
11     FORMAT('p edge', 2I10)
       L = 1
5      I1 = 3*(L-1) + 1
       I2 = I1 + 1
C
       DO 20 I = I1,I2
         WRITE(6,21)I,I+1
   21    FORMAT('e ',2I10,'    1')
         IF(I.LT.I2) WRITE(6,21)I,I+2
20     CONTINUE
C
       IF(L.LT.K)THEN
        L = L + 1
        GO TO 5
       ENDIF
      I=0
      M=0
      DO 100 L = 1, K-1
         M = MOD(M+1, 3)
         IF (M .EQ. 1) THEN
             I = I+1
         ELSE
            I = I + 4
         ENDIF
         WRITE(6,21) I, I+3
  100 CONTINUE
      STOP
      END
