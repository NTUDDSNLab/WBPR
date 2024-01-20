CCCCCCCCCCCCCCCCCCC file: capt.f CCCCCCCCCCCCCCCCCCCCCC
C
      INTEGER*4 ITITLE(20),ISYSI,ISYSO,ISYSP,IDIMACS

      INCLUDE "NETPARM"

C-----------------------------------------------------------------------
C  For a detailed description of this generator see:
C    "A report on the computational behavior of a polynomial time
C       network flow algorithm.", Cornell School of OR/IE Technical
C       Report No. 661 (1985).
C-----------------------------------------------------------------------
C  The generator reads parameters from unit 8 in free format.
C  The record should contain the following fields:
C-----------------------------------------------------------------------
C  Real seed
C      a floating point number between 0 and 1.
C  Number of sources
C      integer
C  Number of sinks
C      integer
C  Flow
C     the (integer) average amount of flow through each source
C  Perturb sources
C     1 implies that source generated flows should be perturbed.
C  Perturb sinks
C     1 implies that sink generated flows should be perturbed.
C  Interval Length
C     the (integer) size of the interval from which flow perturbations a
C     chosen.
C  Cost Bits
C     the number of significant bits in the edge costs.
C  Density of cost bits
C     the (real) probability of setting a particular cost bit to 1.
C  Distribution
C     1-5 specifies the distribution used to generate initialize
C     feasible solutions.
C     Distribution 5 is a sparse version of distribution 1 as reported
C     in the Tech. Rep. referenced above.
C  Density of edges
C     the (real) number of edges generated is density*(Number of sources
C     (Number of sinks). This only is significant for distribution
C     number 5, however the parameter must be included for the
C     other distributions.
C
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      REAL*8 EDGES(NEDGES),SUPPLY(NODES),DEMAND(NODES)
      COMMON /A1/EDGES
      COMMON /ITITLE/ITITLE
      COMMON /IO/ISYSP,ISYSI,ISYSO,IDIMACS
      COMMON /NETWORK/FROM,TO,CST,UPPER,LOWER,RHS
      INTEGER*4 FROM(NEDGES),TO(NEDGES),CST(NEDGES),UPPER(NEDGES),
     C LOWER(NEDGES),RHS(NODES)
      COMMON /PARMS/ISEED,
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,IDIST,
     C SEED,PERLEN,PERCNT,DENS
      INTEGER*4 TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,NSRCE,NSINK,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,ITOT,ISEED,IDIST
      REAL*4
     C SEED,PERLEN,PERCNT,DENS
      COMMON /STATS/RTOT1,RTOT2,RTOT3,RTOT4
      REAL*8 RTOT1,RTOT2,RTOT3,RTOT4
      REAL*8 RSEED
      integer*4 ceil,dceil,floor,dfloor
C  DCL RAND ENTRY(FIXED BINARY(31),FLOAT BINARY(21));
C  DCL 1 ITITLE STATIC EXTERNAL,
C      2 TITLE CHAR(80);

      ISYSI = 8
      ISYSO = 9
      ISYSP = 10
      IDIMACS = 11
      READ(ISYSI,*)RSEED,NSRCE,NSINK,CAPRAN,IPSRCE,IPSINK,
     C               INTLEN,COST,PERCNT,IDIST,
     C               DENS
      WRITE(ISYSP,01)RSEED,NSRCE,NSINK,CAPRAN,IPSRCE,
     C         IPSINK,INTLEN,COST,PERCNT,IDIST
01    FORMAT(F11.9,'NS - ',I9,' ND - ',I9,
     C ' FLOW - ',I9,' IPSRCE - ',I1,' IPSINK - ',I1,/,' INTLEN - ',I4,
     C ' COST - ',I4,
     C ' D - ',F4.2,' IDIST - ',I4)
      WRITE(IDIMACS,02)RSEED,NSRCE,NSINK,CAPRAN,IPSRCE,
     C         IPSINK,INTLEN,COST,PERCNT,IDIST,DENS
02    FORMAT('c ',F11.9,'NS - ',I9,' ND - ',I9,/,
     C 'c FLOW - ',I9,' IPSRCE - ',I1,' IPSINK - ',I1,/,
     C 'c INTLEN - ',I4,
     C ' COST - ',I4,
     C ' D - ',F4.2,' IDIST - ',I4,/,
     C 'c Density - ',F4.2)
      SEED = RSEED
      ISEED = 2**15
      ISEED = 2*(ISEED*ISEED - 1) + 1
      ISEED = ISEED * RSEED
C  PUT STRING(TITLE) EDIT(RSEED,'NS - ',NSRCE,'ND - ',
C      NSINK,'FLOW - ',CAPRAN,'I - ',INTLEN,'C - ',COST,
C      'D - ',PERCNT)
C    (F(11,9),X(1),2(A,F(4),X(1)),A,F(9),X(1),2(A,F(4),X(1)),A,F(4,2));
      NEDGE = NSRCE * NSINK
      ICIRC=0
      INODE=MAX(NSRCE,NSINK)
      CALL       CAPTRN(NSRCE,NSINK,INODE,SUPPLY,DEMAND,ICIRC)
      RETURN
      END

      SUBROUTINE CAPTRN(NSRCE,NSINK,INODE,SUPPLY,DEMAND,ICIRC)

      INCLUDE "NETPARM"

C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------

      COMMON /A1/EDGES(NEDGES)
      REAL*8 EDGES
      COMMON /ITITLE/ITITLE
      INTEGER*4 ITITLE(20)
      COMMON /IO/ISYSP,ISYSI,ISYSO,IDIMACS
      INTEGER*4 ISYSI,ISYSO,ISYSP,IDIMACS
      COMMON /NETWORK/FROM,TO,CST,UPPER,LOWER,RHS
      INTEGER*4 FROM(NEDGES),TO(NEDGES),CST(NEDGES),UPPER(NEDGES),
     C LOWER(NEDGES),RHS(NODES)
C     PARAMETERS COMMON
      COMMON /PARMS/ISEED,
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,IDIST,
     C SEED,PERLEN,PERCNT,DENS
      INTEGER*4
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,
     C NEDGE,ARCCLS,HIGH,LOW,IPSRCE,IPSINK,ITOT,ISEED,IDIST
      REAL*4
     C SEED,PERLEN,PERCNT,DENS
C     STATS COMMON
      COMMON /STATS/RTOT1,RTOT2,RTOT3,RTOT4
      REAL*8 RTOT1,RTOT2,RTOT3,RTOT4
C     LOCAL VARIABLES
      INTEGER*4 NSRCE,NSINK,INODE,
     C SPPLYP,SSRCE,SSINK,
     C PT,IRT,MXIT,IPROB,ICIRC
      REAL*8
     C FRQ,P0,PBAR,dsrce,dsink,dedge,
     C SUPPLY(INODE),
     C DEMAND(NSINK)
C     FUNCTIONS
      integer*4 ceil,dceil,floor,dfloor
      REAL*4 RHOLD
      SSRCE = NSINK + NSRCE + 1
      SSINK = SSRCE + 1
      IPROB = 0
      PT = 0
      IRT = 0
      MXIT = 100000
      FRQ = 8.0
      P0 = 1.0
      PBAR = 1.
      I=NSRCE+NSINK+NSRCE*NSINK+1
C     CALL TITL(SSINK,I)
      DO 01 J = 1,NSINK
          DEMAND(J) = 0
01        CONTINUE
      DO I = 1,NODES
          RHS(I) = 0
      ENDDO
      TOTSUP = 0
      TOTDEM = 0
      RTOT1=0.0
      RTOT2=0.0
      RTOT3=0.0
      RTOT4=0.0
      ITOT=0
      IF (IDIST .EQ. 2) THEN
      DO 101 ISINK = 1 ,  NSINK
          CALL RANDOM(ISEED,SEED)
          SPPLYP = FLOOR(CAPRAN * SEED)
          ITOT=ITOT+SPPLYP
          CALL CREATU(SPPLYP,NSRCE,SUPPLY)
          II = ISINK
          CALL LOAD1(NSRCE,II,NSINK,SUPPLY(1))
101           CONTINUE
      ELSE IF (IDIST .EQ. 3) THEN
          CALL ORDER1(NSRCE,NSINK,SUPPLY)
      ELSE IF (IDIST .EQ. 4) THEN
      ITOT=0
      DO 1041 ISRCE = 1 , NSRCE
          CALL RANDOM(ISEED,SEED)
          SPPLYP = FLOOR(CAPRAN * SEED)
          ITOT=ITOT+SPPLYP
          CALL CREATU(SPPLYP,NSINK,SUPPLY)
          II=(ISRCE-1)*NSINK+1
          CALL LOAD1(NSINK,II,1,SUPPLY(1))
1041          CONTINUE
      DO 1042 ISINK = 1,NSINK
          CALL RANDOM(ISEED,SEED)
          SPPLYP = FLOOR(CAPRAN * SEED)
          ITOT=ITOT+SPPLYP
          CALL CREATU(SPPLYP,NSRCE,SUPPLY)
          II = ISINK
          CALL LOAD1(NSRCE,II,NSINK,SUPPLY(1))
1042          CONTINUE
      ELSE IF (IDIST .EQ. 5) THEN
C         if(1.eq.0) then
C             NEDGE = DENS * NEDGE
C             dsrce = dfloat(nsrce)
C             dsink = dfloat(nsink)
C             do i=1,nedge
C                 call random(iseed,seed)
C                 from(i) = dceil(seed*dsrce)
C                 call random(iseed,seed)
C                 to(i) = dceil(seed*dsink) + nsrce
C             enddo
C         endif
C	oct 1 jc -- handling case of nedge (=nsrce*nsink) >= nedges
 	if (nedge .lt. nedges) then
              do i=1,nedge
                   edges(i)=i
              enddo
              dsrce = dfloat(nsrce)
              dsink = dfloat(nsink)
              dedge = dfloat(nedge)
              KEDGE = DENS * NEDGE
              do i=0,kedge-1
                  call random(iseed,seed)
                  iedge = dceil(seed*dedge)
                  iedge=edges(i+iedge)
C	oct 1 jc -- changing edges(i+iedge) to edges(iedge)
                  edges(iedge)=edges(i+1)
                  edges(i+1)=iedge
                  from(i+1)=1+(iedge-1)/nsink
                  to(i+1)=nsrce+iedge-(from(i+1)-1)*nsink
                  dedge=dedge-1
              enddo
		WRITE(ISYSP,*)'	NO hashing, nedge = ', nedge
              nedge=kedge
	else
C	oct 1 jc -- if nedge too big, use linear hashing to detect duplicates
C			simple, but not too efficient,
C			specially for no. requested arcs > 1/2 nedges
C	set "hashing prime" ihprime to a prime number less than nedges
C	here is a table of useful primes
C	nedges = 1024 2048 4096 8192 16,384 32,768 65,536 131,072
C	ihprime= 1021 2039 4093 8191 16,381 32,749 65,521 131,071
C
C	nedges = 262,144 524,288 550,000 1,048,576 1,200,000
C	ihprime= 262,139 524,287 549,979 1,048,573 1,199,999
		ihprime= 131071
		ihpretry= 0
		if (ihprime .ge. nedges)
     c	WRITE(0,*)'	ERROR: nedges = ',nedges,' too small'
              do i=1,nedges
                   edges(i)=0
              enddo
              dsrce = dfloat(nsrce)
              dsink = dfloat(nsink)
              dedge = dfloat(nedge)
              KEDGE = DENS * NEDGE
              do i=0,kedge-1
520              call random(iseed,seed)
                 iedge = dceil(seed*dedge)
		ihpindex = mod(iedge, ihprime) + 1
530		if ( idnint(edges(ihpindex)) .ne. 0) then
			ihpretry= ihpretry + 1
		if ( idnint(edges(ihpindex)) .eq. iedge ) goto 520
			ihpindex = ihpindex + 1
			if (ihpindex .gt. nedges) ihpindex = 1
			goto 530
		endif
		edges(ihpindex) = dfloat(iedge)
C
                  from(i+1)=1+(iedge-1)/nsink
                  to(i+1)=nsrce+iedge-(from(i+1)-1)*nsink
                  dedge=dedge-1
              enddo
              nedge=kedge
		WRITE(ISYSP,*)'	no. collisions = ', ihpretry
        endif
C
      CALL ORDER5
              do i=1,nedge
          ii = idint(edges(I) * dfloat( nsrce * capran ) )
          IF (INTLEN .NE. 0) THEN
              CALL RANDOM(ISEED,SEED)
              HIGH = ii
     c                 + CEIL(SEED * INTLEN )
          ELSE
              HIGH = ii
          END IF
                      supply(from(i)) = supply(from(i)) +  ii
                      demand(to(i)-nsrce) = demand(to(i)-nsrce) +  ii
          CALL PUTARC(0,HIGH,from(i),to(i),1,1)
      enddo
      END IF
      DO 105 ISRCE = 1 , NSRCE
      IF (IDIST .EQ. 1 .OR. IDIST .EQ. 0) THEN
          CALL RANDOM(ISEED,SEED)
          SPPLYP = FLOOR(CAPRAN * SEED)
          ITOT=ITOT+SPPLYP
          RTOT1=RTOT1+SPPLYP
          RTOT2=RTOT2+FLOAT(SPPLYP)*FLOAT(SPPLYP)
      END IF
      IF ( IDIST .EQ. 4 ) THEN
          CALL CREATY(SPPLYP,SUPPLY,NSINK,ITOT)
      ELSE IF ( IDIST .EQ. 3) THEN
          CALL CREATW(SPPLYP,SUPPLY,NSRCE,NSINK,CAPRAN)
      ELSE IF ( IDIST .EQ.  2) THEN
          CALL CREATY(SPPLYP,SUPPLY,NSINK,ITOT)
      ELSE IF ( IDIST .EQ.  1) THEN
          CALL CREATU(SPPLYP,NSINK,SUPPLY)
      ELSE
                      spplyp = supply(isrce)
      END IF
              if(idist.ne.5) then
      DO 106 ISINK = 1 , NSINK
          IF (INTLEN .NE. 0) THEN
              CALL RANDOM(ISEED,SEED)
              HIGH = SUPPLY(ISINK) + CEIL(SEED * INTLEN )
          ELSE
              HIGH = SUPPLY(ISINK)
          END IF
          CALL PUTARC(0,HIGH,ISRCE,NSRCE+ISINK,1,1)
106           CONTINUE
      DO 150 J =1,NSINK
          DEMAND(J) = DEMAND(J) + SUPPLY(J)
150           CONTINUE
      endif
      TOTSUP = TOTSUP + SPPLYP
      HIGH = SPPLYP
      RTOT3 = RTOT3 + SPPLYP
      RTOT4=RTOT4+FLOAT(SPPLYP)*FLOAT(SPPLYP)
      IF ( IPSRCE .EQ. 1) THEN
          CALL RANDOM(ISEED,SEED)
          IF ( INTLEN .NE. 0) THEN
              HIGH = SPPLYP + CEIL(SEED * INTLEN)
          ELSE
              HIGH = SPPLYP
          END IF
      END IF
      CALL PUTARC(0,HIGH,SSRCE,ISRCE,0,2)
105       CONTINUE
      DO 200 ISINK = 1 , NSINK
      LOW = DEMAND(ISINK)
      IF ( IPSRCE .EQ.  1 .AND. INTLEN .EQ. 0 ) THEN
          IPSINK = 1
      END IF
      IF ( IPSINK .EQ. 1 ) THEN
          CALL RANDOM(ISEED,SEED)
          IF ( INTLEN .NE. 0) THEN
              LOW = MAX0(0,LOW - NINT(INTLEN * SEED))
          END IF
      END IF
      IF ( ICIRC .EQ. 1 ) THEN
          CALL PUTARC(LOW,TOTSUP,NSRCE+ISINK,SSINK,0,3)
      ELSE
          CALL PUTARC(LOW,LOW,NSRCE+ISINK,SSINK,0,3)
      END IF
      TOTDEM = TOTDEM + LOW
200       CONTINUE
      CALL PUTARC(0,TOTSUP,SSINK,SSRCE,0,4)
      IF ( IDIST .EQ. 1 .OR.  IDIST .EQ. 0 ) THEN
      RTOT1 = RTOT1/ITOT
      RTOT2 = RTOT2/ITOT
      RTOT2 = RTOT2/ITOT
      END IF
      RTOT3 = RTOT3/TOTSUP
      RTOT4 = RTOT4/TOTSUP
      RTOT4 = RTOT4/TOTSUP
      RTOT1=RTOT1/NSRCE
      RTOT2=(RTOT2-NSRCE*RTOT1*RTOT1)/(NSRCE-1)
      RTOT2=SQRT(RTOT2)
      RTOT3=RTOT3/NSRCE
      RTOT4=(RTOT4-NSRCE*RTOT1*RTOT1)/(NSRCE-1)
      RTOT4=SQRT(RTOT4)
      WRITE(ISYSP,*) NSRCE,RTOT1,RTOT2
      WRITE(ISYSP,*) NSRCE,RTOT3,RTOT4
      RTOT1=0.0
      RTOT2=0.0
      DO 300 ISINK = 1 , NSINK
         RTOT1=RTOT1+DEMAND(ISINK)/TOTSUP
         RTOT2=RTOT2+(DEMAND(ISINK)/TOTSUP)**2
300      CONTINUE
      RTOT1=RTOT1/NSINK
      RTOT2=(RTOT2-NSINK*RTOT1*RTOT1)/(NSINK-1)
      RTOT2=SQRT(RTOT2)
      WRITE(ISYSP,*) NSINK,RTOT1,RTOT2
      CALL PUTA(2+NSRCE+NSINK,0,NSRCE,NSINK,0,0,1)
      RETURN
      END

      SUBROUTINE CREATW(IR,RARRAY,NN,N,CAPRAN)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      COMMON /STATS/RTOT1,RTOT2,RTOT3,RTOT4
      REAL*8 RTOT1,RTOT2,RTOT3,RTOT4
      REAL*8 RARRAY(NN),R
      INTEGER*4 IR,N,J,CAPRAN
      IR=0
      R=0.0
      CALL DUMP1(N,RARRAY(1))
      DO 100 II=1 , N
      R=R+RARRAY(II)
      RARRAY(II)=DINT(RARRAY(II)*NN*CAPRAN)
      IR=IR+RARRAY(II)
100       CONTINUE
      RTOT1=RTOT1+R
      RTOT2=RTOT2+R*R
      RETURN
      END

      SUBROUTINE CREATY(IR,RARRAY,N,ITOT)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      COMMON /STATS/RTOT1,RTOT2,RTOT3,RTOT4
      REAL*8 RTOT1,RTOT2,RTOT3,RTOT4
      REAL*8 RARRAY(N),R
      INTEGER*4 IR,N,J,ITOT
      IR=0
      R=0.0
      CALL DUMP1(N,RARRAY(1))
      DO 100 II=1 , N
      R=R+RARRAY(II)
      IR=IR+RARRAY(II)
100       CONTINUE
      R=R/ITOT
      RTOT1=RTOT1+R
      RTOT2=RTOT2+R*R
      RETURN
      END

      SUBROUTINE CREATU(TOT1,N,SUPPLY)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      COMMON /PARMS/ISEED,
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,IDIST,
     C SEED,PERLEN,PERCNT
      INTEGER*4
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,NSRCE,NSINK,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,ITOT,ISEED,IDIST
      REAL*4
     C SEED,PERLEN,PERCNT
      REAL*8 SUPPLY(N)
      INTEGER*4 TOT1,N,SUBTOT,J,CEIL
      REAL*8 VL,VH
      REAL*4 TSEED
      SUBTOT = 0
      TSEED = 0.0
      CALL RANDOM(ISEED,SEED)
      J =  CEIL(SEED*N)
      DO 10 I = 1,N
      CALL RANDOM(ISEED,SEED)
      SUPPLY(I) = SEED
      TSEED = TSEED + SEED
10        CONTINUE
      I=1
      NM1 = N - 1
      CALL SORT(I,NM1,N,SUPPLY)
      VL=SUPPLY(1)
      SUPPLY(N)=1.0
      DO 100 I=1 ,  NM1
      VH=SUPPLY(I+1)-SUPPLY(I)
      SUPPLY(I)=VL
      VL=VH
100       CONTINUE
      SUPPLY(N)=VH
      II = 1
      DO 200 I = 1 , NM1
      IF (II .EQ. J) THEN
          II = II + 1
      END IF
      SUPPLY(II) =  DINT(SUPPLY(II) * TOT1)
      SUBTOT = SUBTOT + SUPPLY(II)
      II = II + 1
200       CONTINUE
      SUPPLY(J) = TOT1 - SUBTOT
      RETURN
      END


      SUBROUTINE ORDER1(NSRCE,NSINK,SUPPLY)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------


      INCLUDE "NETPARM"

      COMMON /A1/EDGES(NEDGES)
      REAL*8 EDGES
      COMMON /PARMS/ISEED,
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,IDIST,
     C SEED,PERLEN,PERCNT
      INTEGER*4
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,NSRCE,NSINK,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,ITOT,ISEED,IDIST
      REAL*4
     C SEED,PERLEN,PERCNT
      REAL*8 SUPPLY(NSINK)
      INTEGER*4 NN,NNN,II
      DO 100 ISRCE = 1,NSRCE
      DO 200 ISINK = 1,NSINK
          CALL RANDOM(ISEED,SEED)
          SUPPLY(ISINK)=SEED
200           CONTINUE
      II=(ISRCE-1)*NSINK+1
      CALL LOAD1(NSINK,II,1,SUPPLY(1))
100       CONTINUE
      NN=NEDGE-1
      NNN=1
      CALL SORT(NNN,NN,NN,EDGES)
      CALL ORDER2(NEDGE)
      RETURN
      END

      SUBROUTINE ORDER5


      INCLUDE "NETPARM"

C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      COMMON /A1/EDGES(NEDGES)
      REAL*8 EDGES
      COMMON /PARMS/ISEED,
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,IDIST,
     C SEED,PERLEN,PERCNT
      INTEGER*4
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,NSRCE,NSINK,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,ITOT,ISEED,IDIST
      REAL*4
     C SEED,PERLEN,PERCNT
      INTEGER*4 NN,NNN,II
      DO 100 I = 1,NEDGE
      CALL RANDOM(ISEED,SEED)
      EDGES(I)=SEED
100       CONTINUE
      NN=NEDGE-1
      NNN=1
      CALL SORT(NNN,NN,NN,EDGES)
      CALL ORDER2(NEDGE)
      RETURN
      END


      SUBROUTINE PUTARC(LOW,HIGH,I,II,COSTSW,IARC)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      INTEGER*4 I,II,CST,LOW,HIGH,IARC,COSTSW,ARCCTR,PP
      COMMON /PARMS/ISEED,
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,IDIST,
     C SEED,PERLEN,PERCNT
      INTEGER*4
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,NSRCE,NSINK,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,ITOT,ISEED,IDIST
      ARCCTR = ARCCTR + 1
      IF ( COSTSW .NE. 0) THEN
      PP=1
      CST=0
      DO 10 J=1,COST
          CALL RANDOM(ISEED,SEED)
          CST=CST+PP*(SIGN(1.0,PERCNT-SEED)+1)/2
          PP=2*PP
10            CONTINUE
      ELSE
      CST = 0
      END IF
      CALL PUTA(I,II,CST,HIGH,LOW,IARC,0)
      RETURN
      END

      SUBROUTINE CREATV(TOT1,N,V)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      COMMON /PARMS/ISEED,
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,IDIST,
     C SEED,PERLEN,PERCNT
      INTEGER*4
     C TOTSUP,TOTDEM,COST,CAPRAN,INTLEN,NSRCE,NSINK,
     C NEDGE,ARCCLS,IPSRCE,IPSINK,ITOT,ISEED,IDIST
      REAL*4
     C SEED,PERLEN,PERCNT
      INTEGER*4 TOT1,N,SUBTOT,NM1,CEIL,J
      REAL*8 V(N)
      REAL*4 TSEED
      SUBTOT = 0
      TSEED = 0.0
      CALL RANDOM(ISEED,SEED)
      J =  CEIL(SEED*N)
      DO 100 I = 1 , N
      CALL RANDOM(ISEED,SEED)
      V(I) = SEED
      TSEED = TSEED + SEED
100       CONTINUE
      DO 110 I = 1 ,N
      V(I) = V(I)/TSEED
110       CONTINUE
      II = 1
      DO 200 I = 1 , NM1
      IF (II .EQ. J) THEN
          II = II + 1
      END IF
      V(II) =  DINT(V(II) * TOT1)
      SUBTOT = SUBTOT + V(II)
      II = II + 1
200       CONTINUE
      V(J) = TOT1 - SUBTOT
      RETURN
      END

      SUBROUTINE TITL(I,II)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      COMMON /IO/ISYSP,ISYSI,ISYSO,IDIMACS
      INTEGER*4  ISYSP,ISYSI,ISYSO,IDIMACS

      INCLUDE "NETPARM"


      COMMON /A1/EDGES(NEDGES)
      REAL*8 EDGES
      COMMON /ITITLE/TITLE
      INTEGER TITLE(20)
C     WRITE(ISYSO,10)TITLE
C     WRITE(ISYSO)TITLE
10    FORMAT(20A4)
      IPROB = 0
      PT = 0
      IRT = 0
      MXIT = 100000
      FRQ = 8.0
      P0 = 1.0
      PBAR = 1.5
      III=0
C     WRITE(ISYSO,20)I,III,II,IPROB,PT,IRT,MXIT,FRQ,P0,PBAR
      WRITE(ISYSO)I,III,II,IPROB,PT,IRT,MXIT,FRQ,P0,PBAR
      WRITE(IDIMACS,21)I,III,II,IPROB,PT,IRT,MXIT,FRQ,P0,PBAR
20    FORMAT(3I5,4I8,3F8.1)
21    FORMAT('c',3I5,4I8,3F8.1)
      RETURN
      END

      SUBROUTINE PUTA(IFROM,ITO,ICOST,IHIGH,ILOW,ITYPE,IDUMP)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------


      INCLUDE "NETPARM"

      COMMON /NETWORK/FROM,TO,CST,UPPER,LOWER,RHS
      INTEGER*4 FROM(NEDGES),TO(NEDGES),CST(NEDGES),UPPER(NEDGES),
     C LOWER(NEDGES),RHS(NODES)
      COMMON /IO/ISYSP,ISYSI,ISYSO,IDIMACS
      INTEGER*4  ISYSP,ISYSI,ISYSO,IDIMACS

      INTEGER*4  IFROM,ITO,ICOST,IHIGH,ILOW,ITYPE,IDUMP
      INTEGER*4 IARC,iarcs
      DATA IARC/0/,iarcs/0/
      IF(IDUMP.EQ.1)GOTO01
      IARC = IARC + 1
      FROM(IARC) = IFROM
      TO(IARC)   = ITO
      CST(IARC) = ICOST
      UPPER(IARC) = IHIGH - ILOW
      LOWER(IARC)  = 0
      RHS(IFROM) = RHS(IFROM) - ILOW
      RHS(ITO) = RHS(ITO) + ILOW
      if(ihigh.ne.ilow) then
          iarcs = iarcs+1
      else
          iarcs=iarcs
      endif
      RETURN
01    CONTINUE
10    FORMAT(6X,2I6,2X,3I10)
C-jc	oct 1 jc -- changing I6 to I10
11    FORMAT('p min',6X,2I10)
12    FORMAT('a',6X,2I10,2X,3I10)
C
C     UNFORMATTED WRITE
C
C     WRITE(ISYSO)IFROM,IARC,ICOST,IHIGH
C     WRITE(ISYSO)(FROM(I),I=1,IARC)
C     WRITE(ISYSO)(TO(I),I=1,IARC)
C     WRITE(ISYSO)(COST(I),I=1,IARC)
C     WRITE(ISYSO)(HIGH(I),I=1,IARC)
C     WRITE(ISYSO)(LOW(I),I=1,IARC)
C
C     FORMATTED WRITE
C
C     WRITE(ISYSO,20)IFROM,IARC,ICOST,IHIGH
C     WRITE(ISYSO,20)(FROM(I),I=1,IARC)
C     WRITE(ISYSO,20)(TO(I),I=1,IARC)
C     WRITE(ISYSO,20)(COST(I),I=1,IARC)
C     WRITE(ISYSO,20)(HIGH(I),I=1,IARC)
C     WRITE(ISYSO,20)(LOW(I),I=1,IARC)
20    FORMAT(8(1X,I9))
C     WRITE(ISYSO,10)IFROM,ITO,ICOST,IHIGH
      IPROB = 0
      PT = 0
      IRT = 0
      MXIT = 100000
      FRQ = 8.0
      P0 = 1.0
      PBAR = 1.5
      WRITE(ISYSO)IFROM,0,IARC,ICOST,IHIGH
      WRITE(IDIMACS,11)IFROM,IARCs
      DO I = 1, IFROM
          IF(RHS(I).NE.0) then
              write(IDIMACS,'(''n '',I10,2X,I10)')I,RHS(I)
              if(i.ne.ifrom) then
                  if(rhs(i).gt.0) then
                      WRITE(ISYSO)IFROM,I,0,RHS(I),RHS(I)
                  else
                      WRITE(ISYSO)I,IFROM,0,-RHS(I),-RHS(I)
                  endif
              endif
          endif
      enddo
      DO 100 I=1,IARC
          IF(UPPER(I).GT.0) THEN
C         WRITE(ISYSO,10)FROM(I),TO(I),CST(I),UPPER(I),LOWER(I)
          WRITE(IDIMACS,12)FROM(I),TO(I),LOWER(I),UPPER(I),CST(I)
          WRITE(ISYSO)FROM(I),TO(I),CST(I),UPPER(I),LOWER(I)
          ENDIF
100       CONTINUE
      RETURN
      END
      SUBROUTINE SORT(I,L,N,EDGES)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------


      INCLUDE "NETPARM"

      REAL*8 EDGES(N)
      INTEGER I,L,IRD
      INTEGER II(100),MM(100),LL(100),ICALL(100)
      REAL*8 WORK(NEDGES)
      INTEGER I1,I2,L1,L2,J
C     WRITE(6,*)(EDGES(IV),IV=I,L)
      IRD=1
      ICALL(IRD)=1
      II(IRD)=I
      MM(IRD)=I+(L-I)/2
      LL(IRD)=L
10    CONTINUE
      IC=ICALL(IRD)
      GOTO(100,200,300),IC
100   CONTINUE
      I=II(IRD)
      L=MM(IRD)
      IF(I.EQ.L)GOTO150
      IRD=IRD+1
      ICALL(IRD)=1
      II(IRD)=I
      LL(IRD)=L
      MM(IRD)=I+(L-I)/2
      GOTO10
150   CONTINUE
      ICALL(IRD)=2
200   CONTINUE
      I=MM(IRD)+1
      L=LL(IRD)
      IF(I.EQ.L)GOTO300
      IRD=IRD+1
      ICALL(IRD)=1
      II(IRD)=I
      LL(IRD)=L
      MM(IRD)=I+(L-I)/2
      GOTO10
300   CONTINUE
      I1=II(IRD)
      I2=MM(IRD)
      I3=I2+1
      I4=LL(IRD)
      I5=I1
350   CONTINUE
      IF(I5.GT.I4)GOTO500
      IF(I3.GT.I4)GOTO400
      IF(I1.GT.I2)GOTO360
      IF(EDGES(I1).LT.EDGES(I3))GOTO400
360   CONTINUE
      WORK(I5)=EDGES(I3)
      I3=I3+1
      I5=I5+1
      GOTO350
400   CONTINUE
      WORK(I5)=EDGES(I1)
      I1=I1+1
      I5=I5+1
      GOTO350
500   CONTINUE
      I1=II(IRD)
      I4=LL(IRD)
      DO 510 I5=I1,I4
      EDGES(I5)=WORK(I5)
510       CONTINUE
      IRD=IRD-1
      IF(IRD.EQ.0)RETURN
      ICALL(IRD)=ICALL(IRD)+1
      GOTO10
      END
      SUBROUTINE LOAD1(N,II,IS,SUPPLY)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------

      INCLUDE "NETPARM"


      COMMON /A1/EDGES(NEDGES)
      REAL*8 EDGES
      REAL*8 SUPPLY(N)
      INTEGER IX
      DATA IX/0/
      IF(IX.NE.0)GOTO9
      DO 5 I=1,NEDGES
      EDGES(I)=0.0
5         CONTINUE
       IX = 1
9      CONTINUE
       DO 10 I=1,N
       EDGES(II)=EDGES(II)+SUPPLY(I)
      II=II+IS
C         WRITE(6,*)EDGES(I)
10        CONTINUE
      RETURN
      END
      SUBROUTINE DUMP1(N,SUPPLY)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------

      INCLUDE "NETPARM"


      COMMON /A1/EDGES(NEDGES)
      REAL*8 EDGES
      REAL*8 SUPPLY(N)
      INTEGER I
      DATA I/0/
      DO 10 II=1,N
C         WRITE(6,*)I,N,SUPPLY
      I=I+1
      SUPPLY(II)=EDGES(I)
10        CONTINUE
      RETURN
      END
      SUBROUTINE ORDER2(N)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------

      INCLUDE "NETPARM"


      COMMON /A1/EDGES(NEDGES)
      REAL*8 EDGES
      EDGES(N)=1.0
      II=N-1
      DO 10 I=1,II
      EDGES(N-I+1)=EDGES(N-I+1)-EDGES(N-I)
10        CONTINUE
      RETURN
      END

      SUBROUTINE RANDOM(ISEED,SEED)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      INTEGER*4 ISEED
      REAL*4 SEED
      REAL*8 RSEED
      CALL RAND_LOCAL(ISEED,RSEED)
      SEED = RSEED
C     WRITE(15,90)ISEED,SEED
90    FORMAT('RANDOM NUMBER -',I12,1X,F12.10)
      RETURN
      END

      INTEGER FUNCTION FLOOR*4(RNUM)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      REAL*4 RUM
      FLOOR = INT(AINT(RNUM))
      RETURN
      END

      INTEGER FUNCTION DFLOOR*4(RNUM)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      REAL*8 RNUM
      DFLOOR = DINT(RNUM)
      RETURN
      END

      INTEGER FUNCTION DCEIL*4(RNUM)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      REAL*8 RNUM,RR
      RR = DFLOAT(IDINT(RNUM))
      IF(RNUM.EQ.DFLOAT(IDINT(RNUM))) THEN
      DCEIL = IDINT(RNUM)
      ELSE
      DCEIL = 1 + IDINT(RNUM)
      END IF
      RETURN
      END

      INTEGER FUNCTION CEIL*4(RNUM)
C-----------------------------------------------------------------------
C     c 1991 copyright Robert G. Bland and David L. Jensen
C                      Ithaca, New York
C            All rights reserved
C-----------------------------------------------------------------------
      REAL*4 RNUM
      IF(RNUM.EQ.AINT(RNUM)) THEN
      CEIL = INT(AINT(RNUM))
      ELSE
      CEIL = 1 + INT(AINT(RNUM))
      END IF
      RETURN
      END
       subroutine rand_local(iseed,seed)
       integer*4 iseed
       real*8 seed
       iseed = iabs(iseed * 16807)
       seed = dfloat(iseed) / dfloat(2*(2**30-1) +1)
       return
       end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

