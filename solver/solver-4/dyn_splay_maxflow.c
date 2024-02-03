/* dyn_splay_maxflow.c ==== rotate subroutines for the splay step 
   in dynamic tree data structure */

/* Copyright:
	This program was written by 

	Tamas Badics, 1991,
	Rutgers University, RUTCOR
	P.O.Box 5062
	New Brunswick, NJ, 08903
	e-mail: badics@rutcor.rutgers.edu
 
	The code may be used and modified for not-for-profit use.
	This notice must be remained.	
====================================================================*/

#include <stdio.h>
#include "_dyn_tree_maxflow.h"
#include "macros.h"

/*==================================================================*/
static void __dyn_step_r( vertex * f, vertex * ch)
{
	DOUBLE tmp, tmp1;
	vertex * b, * c, *chf;
	
	b = ch->dleft;
	c = f->dleft;
	
	if( NULL != (f->dright = b) )
	  b->dfather = f;
	ch->dleft = f;
	
	chf = ch->dfather = f->dfather;
	if( chf->dright == f)
	  chf->dright = ch;
	else if(chf->dleft == f)
	  chf->dleft = ch;
	
	f->dfather = ch;
	
	/* Value update: */
	
	tmp = dvh;
	if (b)
	  dvb += tmp;
	dmh = tmp + dmf;
	dvh += dvf;
	dvf = -tmp;

	tmp = (b ? dmb - dvb : 0);
	tmp1 = (c ? dmc - dvc : 0);
	dmf = MAX3( 0, tmp, tmp1);
}

/*==================================================================*/
static void __dyn_step_l( vertex	* f, vertex * ch)
{
	DOUBLE tmp, tmp1;
	vertex * b, * c, * chf;
	
	b = ch->dright;
	c = f->dright;
	
	if(NULL != (f->dleft = b) )
	  b->dfather = f;
	ch->dright = f;
	
	chf = ch->dfather = f->dfather;
	if( chf->dleft == f)
	  chf->dleft = ch;
	else if(chf->dright == f)
	  chf->dright = ch;
	
	f->dfather = ch;
	
	/* Value update: */
	
	tmp = dvh;
	if (b)
	  dvb += tmp;
	dmh = tmp + dmf;
	dvh += dvf;
	dvf = -tmp;
	
	tmp = (b ? dmb - dvb : 0);
	tmp1 = (c ? dmc - dvc : 0);
	dmf = MAX3( 0, tmp, tmp1);
}

/*==================================================================*/
static void __dyn_step_rl(vertex * g, vertex * f, vertex * ch)
{
	DOUBLE tmp, tmp1;
	vertex * a, * b, * c, * d, * chf;
	
	a = f->dright;
	b = ch->dright;
	c = ch->dleft;
	d = g->dleft;
	
	if( NULL != (g->dright = c))
	  g->dright->dfather = g;
	
	if (NULL != (f->dleft = b))
	  f->dleft->dfather = f;
	ch->dleft	 = g;
	ch->dright	 = f;
	chf = ch->dfather = g->dfather;
	
	if( chf->dright == g)
	  chf->dright = ch;
	else if(chf->dleft == g)
	  chf->dleft = ch;
	
	g->dfather = f->dfather = ch;
	
	/*======== Value update: ========*/
	
	tmp = dvh + dvf;
	if (b)
	  dvb += dvh;
	if (c)
	  dvc += tmp;
	dvf  = -dvh;
	dvh = tmp + dvg;
	dvg  = -tmp;
	dmh = dmg + tmp;

	tmp = (d ? dmd - dvd : 0);
	tmp1 = (c ? dmc - dvc : 0);
	dmg = MAX3( 0, tmp, tmp1);

	tmp = (a ? dma - dva : 0);
	tmp1 = (b ? dmb - dvb : 0);
	dmf = MAX3( 0, tmp, tmp1);
}

/*==================================================================*/
static void __dyn_step_lr(vertex * g, vertex * f, vertex * ch)
{
	DOUBLE tmp, tmp1;
	vertex * a, * b, * c, * d, * chf;
	
	a = f->dleft;
	b = ch->dleft;
	c = ch->dright;
	d = g->dright;
	
	if (NULL != (g->dleft = c) )
	  g->dleft->dfather = g;
	if( NULL != (f->dright = b) )
	  f->dright->dfather = f;
	ch->dright	 = g;
	ch->dleft	 = f;
	chf = ch->dfather = g->dfather;
	
	if( chf->dright == g)
	  chf->dright = ch;
	else if(chf->dleft == g)
	  chf->dleft = ch;
	
	g->dfather = f->dfather = ch;
	
	/*======== Value update: ========*/
	
	tmp = dvh + dvf;
	if (b)
	  dvb += dvh;
	if (c)
	  dvc += tmp;
	dvf  = -dvh;
	dvh = tmp + dvg;
	dvg  = -tmp;
	dmh = dmg + tmp;

	tmp = (d ? dmd - dvd : 0);
	tmp1 = (c ? dmc - dvc : 0);
	dmg = MAX3( 0, tmp, tmp1);

	tmp = (a ? dma - dva : 0);
	tmp1 = (b ? dmb - dvb : 0);
	dmf = MAX3( 0, tmp, tmp1);
}

/*==================================================================*/
static void __dyn_step_rr( vertex * g, vertex * f, vertex * ch)
{
	DOUBLE tmp, tmp1, tmp2, tmp3;
	vertex * b, * c, * d, * chf;
	
	b = ch->dleft;
	c = f->dleft;
	d = g->dleft;
	
	if( NULL != (g->dright = c) )
	  g->dright->dfather = g;
	f->dleft	 = g;
	if( NULL != (f->dright = b))
	  f->dright->dfather = f;
	ch->dleft = f;
	chf = ch->dfather = g->dfather;
	
	if( chf->dright == g)
	  chf->dright = ch;
	else if(chf->dleft == g)
	  chf->dleft = ch;
	ch->dfather	 = g->dfather;
	
	g->dfather = f;
	f->dfather = ch;
	
	/*======== Value update: ========*/
	
	if (b)
	  dvb += dvh;
	if (c)
	  dvc += dvf;
	tmp = dvh + dvf;
	dmh = tmp + dmg;

	tmp2 = (c ? dmc - dvc : 0);
	tmp3 = (d ? dmd - dvd : 0);
	dmg = MAX3( 0, tmp2, tmp3);

	tmp2 = (b ? dmb - dvb : 0);
	tmp3 = dvf + dmg;
	dmf = MAX3( 0, tmp2, tmp3);
	tmp1 = -dvh;
	dvh = tmp + dvg;
	dvg = -dvf;
	dvf = tmp1;
}

/*==================================================================*/
static void __dyn_step_ll( vertex * g, vertex * f, vertex * ch)
{
	DOUBLE tmp, tmp1, tmp2, tmp3;
	vertex * b, * c, * d, * chf;
	
	b = ch->dright;
	c = f->dright;
	d = g->dright;
	
	if( NULL != (g->dleft = c) )
	  g->dleft->dfather = g;
	f->dright	 = g;
	if (NULL != (f->dleft = b) )
	  f->dleft->dfather = f;
	ch->dright	 = f;
	chf = ch->dfather = g->dfather;
	
	if( chf->dright == g)
	  chf->dright = ch;
	else if(chf->dleft == g)
	  chf->dleft = ch;
	ch->dfather	 = g->dfather;
	
	g->dfather = f;
	f->dfather = ch;
	
	/*======== Value update: ========*/
	
	if (b)
	  dvb += dvh;
	if (c)
	  dvc += dvf;
	tmp = dvh + dvf;
	dmh = tmp + dmg;

	tmp2 = (c ? dmc - dvc : 0);
	tmp3 = (d ? dmd - dvd : 0);
	dmg = MAX3( 0, tmp2, tmp3);

	tmp2 = (b ? dmb - dvb : 0);
	tmp3 = dvf + dmg;
	dmf = MAX3( 0, tmp2, tmp3);

	tmp1 = -dvh;
	dvh = tmp + dvg;
	dvg = -dvf;
	dvf = tmp1;
}

/*==================================================================*/
void splice(vertex * m)   /* m is a solid root, middle-child 
							   of its f, who is also a solid root.
							   Change m to be the left-child. 
							   (the old left-child will be a middle)*/
{
	DOUBLE tmp, tmp1;
	vertex * f = m->dfather;
	vertex * l = f->dleft;
	vertex * r = f->dright;
	
	f->dleft = m;  /* Changing the solid edge f-l to f-m */
	
	/* Here comes the value update */
	
	dvm -= dvf;
	if (l)
	  dvl += dvf;

	tmp = (r ? dmr - dvr : 0);
	tmp1 = dmm - dvm;
	dmf = MAX3( 0, tmp, tmp1);
}

/*==================================================================*/
void dyn_splay_solid(vertex * ch )	  /* Brings up the ch in its 
										 solid subtree. After this 
										 procedure ch will be the 
										 root of the solid tree.
										 
										 ch cannot be The Root!*/
	 
{
	vertex * g ;
	vertex * f ;
	for(;;){
		f = ch->dfather;
		g = f->dfather;
		
		if (f->dleft == ch){
			if (g->dleft == f)
			  __dyn_step_ll(g, f, ch);
			else if (g->dright == f)
			  __dyn_step_rl(g, f, ch);
			else
			  __dyn_step_l(f, ch);
		}else if (f->dright == ch){
			if (g->dleft == f)
			  __dyn_step_lr(g, f, ch);
			else if (g->dright == f)
			  __dyn_step_rr(g, f, ch);
			else
			  __dyn_step_r(f, ch);
		}else
		  break;
	}
}

/*==================================================================*/
vertex * dyn_splay(vertex * ch )   /* Brings up the ch in its virtual 
									  tree. After this procedure, ch 
									  will be a mid-child of the root 
									  of the virtual tree*/	 
{
	vertex * f;
	
	
	if (ch->dfather == NULL)   /* If ch is The Root */
	  return ch;
	
	dyn_splay_solid(ch);
	
	while ((f = ch->dfather)->dfather){  /* While f is not The Root */
		
		dyn_splay_solid(f);	  /* After this, ch is a middle-child 
								 of its f, and f is
								 a solid root */
		splice(ch);
		__dyn_step_l(f, ch);
	}
	return (ch);
}
