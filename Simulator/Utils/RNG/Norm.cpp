/**
 * @file Norm.cpp
 * 
 * @ingroup Simulator/Utils/RNG
 * 
 * @brief Normally distributed random numbers
 */

/************************************************************
   norm.cpp -- normally distibuted random numbers

   The following class allows you to create MTRand objects that are
   independently seeded upon construction and return normally
   distributed random numbers.

       This function generates normally distributed random numbers
   with mean of mu and standard deviation of sigma, using the
   polar method of Marsaglia and Bray, "A Convenient Method for
   Generating Normal Variables", _SIAM Rev._, 6: 260-264 (1964).

       The algorithm is as follows:

   1. Generate two uniformly distributed numbers, U1 and U2.
      Let Vi = 2*Ui-1 for i=1,2, and let W = V1*V1 + V2*V2.
   2. If W > 1, go back to step 1.  Otherwise, let
      Y = sqrt(-2*ln(W)/W), X1 = V1*Y, and X2 = V2*Y.  Then
      X1 and X2 are normally distributed with mean of 0 and
      variance of 1.
   3. Random numbers with mean of mu and standard deviation of sigma
      are calculated by: X_prime = mu + sigma   X.

   Note that numbers are generated in pairs.  On odd-numbered calls
   to operator(), pairs are calculated.  On even-numbered calls,
   the second value is returned.

   Modified from norm.c, from xneuron3

************************************************************/

// $Log: norm.cpp,v $
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.3  2005/03/08 19:56:31  stiber
// Modified comments for Doxygen.
//
// Revision 1.2  2005/02/18 13:41:51  stiber
// Added SourceVersions support.
//
// Revision 1.1  2005/02/09 18:46:31  stiber
// Initial revision
//
//

#include "Norm.h"

using namespace std;

/*!
   This method makes instances functors; it returns normally
   distributed random numbers. Just a cute way of doing things.
   @return pseudorandom number drawn from a normal distribution.
*/
BGFLOAT Norm::operator()()
{
   BGFLOAT U1, U2,    // Uniformly distributed.
      V1, V2, W, Y,   //Work variables (see above).
      X1;             // First value computed (returned immediately)

   // Check to see if we need to compute anything, complement indicator.
   if ((odd_ = !odd_))
      return (mu_ + sigma_ * X2_);

   // Do the computation step 1 (until W <= 1)
   do {
      U1 = MTRand::operator()(); /* Generate U(0,1) */
      U2 = MTRand::operator()();
      V1 = 2 * U1 - 1;
      V2 = 2 * U2 - 1;
      W = V1 * V1 + V2 * V2;
   } while (W > 1);

   // Do the computation step 2
   Y = sqrt(-2 * log(W) / W);
   X1 = V1 * Y;
   X2_ = V2 * Y;

   // Return X1 this time, X2 next time
   return (mu_ + sigma_ * X1);
}

Norm::Norm(BGFLOAT m, BGFLOAT s, uint32_t seed) : MTRand(seed), odd_(true), mu_(m), sigma_(s)
{
}

void Norm::seed(BGFLOAT m, BGFLOAT s, uint32_t seed)
{
   MTRand::seed(seed);
   odd_ = true;
   mu_ = m;
   sigma_ = s;
}

void Norm::seed(uint32_t seed)
{
   Norm::seed(DEFAULT_m, DEFAULT_s, seed);
}

void Norm::seed()
{
   Norm::seed(DEFAULT_m, DEFAULT_s, DEFAULT_seed);
}