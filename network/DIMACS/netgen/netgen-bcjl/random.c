/*** This is a portable random number generator whose origins are
 *** unknown.  As far as can be told, this is public domain software.


/*** portable random number generator */

/*** Note that every variable used here must have at least 31 bits
 *** of precision, exclusive of sign.  Long integers should be enough.
 *** The generator is the congruential:  i = 7**5 * i mod (2^31-1).
 ***/

#define MULTIPLIER 16807
#define MODULUS    2147483647

static long saved_seed;


/*** set_random - initialize constants and seed */

void set_random(seed)
long seed;
{
  saved_seed = seed;
}


/*** random - generate a random integer in the interval [a,b] (b >= a >= 0) */

long random(a, b)
long a, b;
{
  register long hi, lo;

  hi = MULTIPLIER * (saved_seed >> 16);
  lo = MULTIPLIER * (saved_seed & 0xffff);
  hi += (lo>>16);
  lo &= 0xffff;
  lo += (hi>>15);
  hi &= 0x7fff;
  lo -= MODULUS;
  if ((saved_seed = (hi<<16) + lo) < 0)
    saved_seed += MODULUS;

  if (b <= a)
    return b;
  return a + saved_seed % (b - a + 1);
}
