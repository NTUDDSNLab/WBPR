C  Note: field sizes increased for DIMACS challenge (C. McGeoch, 7/91) 
C     PROGRAM GABOW
C
C
C     REFERENCE:   H. N. GABOW, "AN EFFICIENT IMPLEMENTATION OF EDMONDS'
C                  ALGORITHM FOR MAXIMUM MATCHING ON GRAPHS", JACM 23
C                  (1976), PP. 221-234.
C
C     INPUT:       NONE.  SET THE VALUE OF K IN THE PARAMETER STATEMENT.
C
C     OUTPUT:      NUMBER OF VERTICES = N = 6*K
C                  NUMBER OF EDGES    = M = 8*K*K
C                  LIST OF EDGES (ALL EDGE COSTS ARE SET TO 1)
C
C                  THE GRAPH IS CONSTRUCTED SO THAT VERTICES 1-4K FORM A
C                  COMPLETE SUBGRAPH.  FOR I = 1 TO 2K, VERTEX (2I-1) IS
C                  JOINED TO VERTEX 4K+1.
C
C     PROGRAMMER:  R. BRUCE MATTINGLY
C     LANGUAGE:    FORTRAN 77
C     DATE:        FEBRUARY 28, 1991
C
      PARAMETER (K= 3, N=6*K, M=8*K*K, K4=4*K, K4P = K4+1, K4M = K4-1)
C
      WRITE(6, 7)
    7 FORMAT('c This problem belongs to the class of worst-case graphs',
     + /,'c described by Gabow (JACM 23, pp.221-234)',
     + /,'c Submitted by R. B. Mattingly')
      WRITE(6,11) N, M
   11 FORMAT('p edge', I8, I9)
c  11 FORMAT('p edge',I4,I6)
      DO 20 I = 1, K4M
         DO 30 J = I+1, K4
            WRITE(6, 31) I, J
  31       FORMAT('e',2I8,'   1')
c 31       FORMAT('e',2I4,'   1')
   30    CONTINUE
         IF (MOD(I,2) .EQ. 1) THEN
            J = K4 + (I+1)/2
            WRITE(6, 31) I, J
         ENDIF
   20 CONTINUE
      STOP
      END
