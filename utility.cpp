/*
 * MIT License
 *
 *
 * Copyright (c) 2025 Kun Chen <kunchen@siom.ac.cn>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *
 */

#include <cmath>
#include <stdio.h>
#include "utility.h"

#ifdef __IBH__
// cofficients for the Integrated Blackman-Harris window function
  #define A0  1.0
  #define A1 -1.3611
  #define A2  0.3938
  #define A3 -0.0326
#else
// cofficients for the newly developped 4th-order smooth function
// Refer to paper arXiv:2403.04053 for details.
  #define A0  1.0
  #define A1 -4.0/3.0
  #define A2  1.0/3.0
  #define A3  0.0
#endif

QPrecision ksi(QPrecision x)
{
   if (x<TINY)
      return 0.0;

   if (x>1.0-TINY)
      return 1.0;

   return (A0*x + (A1/TWOPI)*SIN(TWOPI*x) 
	 + (A2/FOURPI)*SIN(FOURPI*x)
	 + (A3/SIXPI)*SIN(SIXPI*x));
}

QPrecision dksi(QPrecision x)
{
   if (x<TINY || x>1.0-TINY)
      return 0.0;

   return (A0 + A1*COS(TWOPI*x)
	 + A2*COS(FOURPI*x)
	 + A3*COS(SIXPI*x));
}

QPrecision d2ksi(QPrecision x)
{
   if (x<TINY || x>1.0-TINY)
      return 0.0;

   return (-TWOPI*(A1*SIN(TWOPI*x)
	    + 2.0*A2*SIN(FOURPI*x)
	    + 3.0*A3*SIN(SIXPI*x)));
}

void Derivative_K_Coeff_v2(QPrecision * k1, QPrecision * k2, QPrecision dD, long nD)
{
   for (long i=0; i<nD; ++i) {
      QPrecision kk;
      if (i<=(nD>>1))
	 kk=TWOPI/dD*(QPrecision) i/(QPrecision) nD;
      else
	 kk=TWOPI/dD*((QPrecision) i/(QPrecision) nD - (QPrecision) 1);

      k1[i*2]=-kk*SIN(kk*dD/2.0);
      k1[i*2+1]=kk*COS(kk*dD/2.0);
      k2[i]=kk*kk;
   }
}

QPrecision weight(QPrecision x)	// the same as ksi; could be defined in other ways
{
   if (x<TINY)
      return 0.0;

   if (x>1.0-TINY)
      return 1.0;

   return (A0*x + (A1/TWOPI)*SIN(TWOPI*x) 
	 + (A2/FOURPI)*SIN(FOURPI*x)
	 + (A3/SIXPI)*SIN(SIXPI*x));
}

bool getaline(std::ifstream& ifs, std::string& aline)
{
   aline.clear();

   while (!ifs.eof() && std::getline(ifs, aline)
	 && ((aline[0]=='#' || aline[0]=='%' || aline[0]=='/')	// a comment line
	 || aline.find_first_not_of(" \t")>aline.length()))	// a blank line
   {
      aline.clear();
   }

   return !aline.empty();
}
