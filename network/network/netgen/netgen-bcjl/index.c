/*** Copyright 1989 Norbert Schlenker.  All rights reserved.

 *** This software is distributed subject to the following provisions:
 ***    - this notice may not be removed;
 ***    - you may modify the source code, as long as redistributed
 ***      versions have their modifications clearly marked;
 ***    - no charge, other than a nominal copying fee, may be made
 ***      when providing copies of this source code to others;
 ***    - if this source code is used as part of a product which is
 ***      distributed only as a binary, a copy of this source code
 ***      must be included in the distribution.
 ***
 *** Unlike the GNU GPL, use of this code does not obligate you to
 *** disclose your own proprietary source code.

 *** The author of this software provides no warranty, express or
 *** implied, as to its utility or correctness.  That said, reports
 *** of bugs or compatibility problems will be gladly received by
 *** nfs@princeton.edu, and fixes will be attempted.
 ***/


/*** index.c - routines to manipulate index lists */

/*** Definition:  An "index list" is an ascending sequence of positive
 ***              integers that can be operated upon as follows:
 ***
 ***                 make_index_list - makes an index list of consecutive
 ***                    integers from some lower bound through an upper bound.
 ***                 choose_index - given a number "k", removes the integer
 ***                    in the k'th position in the index list and returns it.
 ***                    Requests for positions less than 1 or greater than
 ***                    the current list length return zero.
 ***                 remove_index - removes a specified integer from the
 ***                    index list.  Requests to remove integers not in the
 ***                    list are ignored, except that they reduce the list's
 ***                    "pseudo_size" (see below).
 ***                 index_size - returns the number of integers in the
 ***                    index list.
 ***                 pseudo_size - returns the number of integers in the
 ***                    index list, less the number for which remove_index
 ***                    failed due to a request to remove a non-existent one.
 ***			(Note that this is here solely to support an apparent
 ***			bug in the original definition of the NETGEN program.)

 *** Two simple methods of accomplishing these functions are:
 ***   - allocating an array of flags that indicate whether a particular
 ***     integer is valid, and searching the array during the choose_index
 ***     operation for the k'th valid integer.
 ***   - allocating a linked list for the indices and updating the list
 ***     during both the choose_index and remove_index operations.
 ***
 *** For small index lists, the first of these methods is quite efficient
 *** and is, in fact, implemented in the following code.  Unfortunately,
 *** for the uses we have in mind (i.e. NETGEN), the typical access pattern
 *** to index lists involves a large list, with both choose_index and
 *** remove_index operations occurring at random positions in the list.
 ***
 *** As a result, the code has been extended for the case of large lists.
 *** The enclosed implementation makes use of a binary interval tree, which
 *** records information regarding the valid intervals from which indices
 *** may be chosen.  At a cost of a substantial increase in space requirements,
 *** and under rather generous assumptions regarding the randomness of the
 *** positions supplied to choose_index, running time becomes logarithmic
 *** per choose_index and remove_index operation.
 ***/

#include "netgen.h"

/*** Useful constants */
#define FLAG_LIMIT 100		/* upper limit for simple implementation */


/*** Internally useful types */
typedef unsigned char FLAG;

typedef struct index_header {
  INDEX original_size;		/* original size of index */
  INDEX index_size;		/* number of indices in the index */
  INDEX pseudo_size;		/* almost the number of indices in the index */
  union {
    INDEX index_base;		/* base of index list - small case */
    INDEX index_nodes;		/* number of nodes in the interval tree - large case */
  } i;
  union {
    FLAG* flag;			/* pointer to flag array - small */
    struct interval_node* first_node; /* pointer to root of interval tree - large */
  } p;
} HEADER;

typedef struct interval_node {
  INDEX base;			/* smallest integer in this subtree */
  INDEX count;			/* count of indices in this subtree */
  struct interval_node* left_child; /* pointers down the tree */
} INODE;


/*** Static storage */

static INDEX_LIST active_handles = 0;
static HEADER* index_headers = NULL;


/*** Make a new index list with a specified range.  Returns an integer handle
 *** to identify the list, or -1 if an error occurs.
 ***/
INDEX_LIST make_index_list(from, to)
INDEX from;			/* lower limit of index list */
INDEX to;			/* upper limit of index list */
{
  INDEX_LIST handle = 0;
  HEADER* hp;
  INODE* np;

  if (from <= 0 || from > to)	/* sanity check */
    return -1;

/*** Find an inactive list header or allocate a new one. */
  for (hp = index_headers; handle < active_handles; hp++, handle++) {
    if (hp->original_size == 0)
      break;
  }
  if (handle == active_handles) {
    ++active_handles;
    if (handle == 0)
      index_headers = (HEADER*) malloc(active_handles * sizeof(HEADER));
    else
      index_headers = (HEADER*) realloc(index_headers, active_handles * sizeof(HEADER));
  }
  if (index_headers == NULL)
    return -1;


/*** Fill in the list header and allocate space for the list. */
  hp = &index_headers[handle];
  hp->pseudo_size = hp->index_size = hp->original_size = to - from + 1;
  if (hp->original_size <= FLAG_LIMIT) { /* SMALL */
    hp->i.index_base = from;
    hp->p.flag = (FLAG*) malloc(hp->original_size * sizeof(FLAG));
    if (hp->p.flag == NULL)
      return -1;
    (void)memset((void *)hp->p.flag, 0, hp->original_size * sizeof(FLAG));
  } else {			/* LARGE */
    hp->i.index_nodes = 1;
    np = (INODE*) malloc(hp->original_size * sizeof(INODE));
    if (np == NULL)
      return -1;
    hp->p.first_node = np;
    np->base = from;
    np->count = hp->original_size;
    np->left_child = NULL;
  }
  return handle;
}


/*** Free an existing index list.  The header is zeroed afterwards
 *** to indicate that it can be reused.
 ***/
void free_index_list(handle)
INDEX_LIST handle;
{
  HEADER* hp;

  if (handle < 0 || handle >= active_handles)	/* sanity check */
    return;

  hp = &index_headers[handle];
  if (hp->p.flag)
    free((void *)hp->p.flag);
  (void)memset((void *)hp, 0, sizeof(HEADER));
}

/*** Choose the integer at a certain position in an index list.  The
 *** integer is then removed from the list so that it won't be chosen
 *** again.  Choose_index returns 0 if the position is invalid.
 ***/
INDEX choose_index(handle, position)
INDEX_LIST handle;
INDEX position;
{
  HEADER* hp;
  FLAG* cp;
  INODE* np;
  INODE* npl;
  INODE* npr;
  INDEX index;

  if (handle < 0 || handle >= active_handles)	/* sanity checks */
    return 0;
  hp = &index_headers[handle];
  if (hp->p.flag == NULL)
    return 0;
  if (position < 1 || position > hp->index_size)
    return 0;

/*** Adjust counts of remaining indices. */
  hp->index_size--;
  hp->pseudo_size--;


/*** Find the index we want and remove it from the list. */
  if (hp->original_size <= FLAG_LIMIT) { /* SMALL */
    for (cp = hp->p.flag; position > 0; cp++)
      if (!*cp)
	position--;
    *(--cp) = 1;
    return hp->i.index_base + (INDEX)(cp - hp->p.flag);
  } else {			/* LARGE */
    np = hp->p.first_node;
    while (np->left_child) {
      np->count--;
      np = np->left_child;
      if (position > np->count) {
	position -= np->count;
	np++;
      }
    }
    np->count--;
    if (position == 1) {	/* beginning of interval */
      index = np->base++;
    }
    else
    if (position > np->count) {	/* end of interval */
      index = np->base + np->count;
    }
    else			/* middle of interval - split it */
    {
      index = np->base + position - 1;
      npl = np->left_child = hp->p.first_node + hp->i.index_nodes;
      npr = npl + 1;
      hp->i.index_nodes += 2;
      npl->base = np->base;
      npl->count = position - 1;
      npl->left_child = NULL;
      npr->base = index + 1;
      npr->count = np->count - npl->count;
      npr->left_child = NULL;
    }
    return index;
  }
}

/*** Remove a particular integer from an index list.  If the integer
 *** does not exist in the list, we reduce the list's pseudo-size,
 *** but return no error indication.
 ***/
void remove_index(handle, index)
INDEX_LIST handle;
INDEX index;
{
  HEADER* hp;
  FLAG* cp;
  INODE* np;
  INODE* npl;
  INODE* npr;

  if (handle < 0 || handle >= active_handles)	/* sanity checks */
    return;
  hp = &index_headers[handle];
  if (hp->p.flag == NULL)
    return;

/*** Adjust the pseudo-size before looking for the index. */
  hp->pseudo_size--;

/*** Remove the index from the index list. */
  if (hp->original_size <= FLAG_LIMIT) { /* SMALL */
    if (index < hp->i.index_base || index >= hp->i.index_base + hp->original_size)
      return;
    cp = hp->p.flag + (index - hp->i.index_base);
    if (!*cp) {
      *cp = 1;
      hp->index_size--;
    }
    return;
  } else {			/* LARGE */
    np = hp->p.first_node;
    while (np->left_child) {
      np->count--;
      np = np->left_child + 1;
      if (index < np->base)
	np--;
    }
    if (index < np->base || index >= np->base + np->count) { /* mistake - back out */
      np = hp->p.first_node;
      while (np->left_child) {
        np->count++;
	np = np->left_child + 1;
	if (index < np->base)
	  np--;
      }
      return;
    }
    np->count--;
    if (index == np->base) {			/* beginning of interval */
      np->base++;
    }
    else
    if (index == np->base + np->count) {	/* end of interval */
    }
    else    	    	    			/* middle of interval - split it */
    {
      npl = np->left_child = hp->p.first_node + hp->i.index_nodes;
      npr = npl + 1;
      hp->i.index_nodes += 2;
      npl->base = np->base;
      npl->count = index - np->base;
      npl->left_child = NULL;
      npr->base = index + 1;
      npr->count = np->count - npl->count;
      npr->left_child = NULL;
    }
    hp->index_size--;
    return;
  }
}


/*** Return actual number of remaining entries in the index list.
 ***/
INDEX index_size(handle)
INDEX_LIST handle;
{
  if (handle < 0 || handle >= active_handles)	/* sanity check */
    return 0;

  return index_headers[handle].index_size;
}


/*** Return a peculiar number, vaguely related to the number of
 *** remaining entries in the index list.  Required by NETGEN.
 ***/
INDEX pseudo_size(handle)
INDEX_LIST handle;
{
  if (handle < 0 || handle >= active_handles)	/* sanity check */
    return 0;

  return index_headers[handle].pseudo_size;
}
