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
#include "run_environment.h"

// The value of $\gamma\mu_N\mu_0 M_0$  is set to 50%
// the neutron energy, the radius of magnetic sphere is set
// to the neutron wavelength, i.e. 2\pi\lambdabar,
// the magnetization of the uniformly magnetized sphere is
// along the y-axis.
#define V0	0.327216805*0.5
#define RADIUS	6.283185307179586476925286766559005768394
#define RADIUS3	RADIUS*RADIUS*RADIUS

void Potential(QPrecision *VB, QPrecision x, QPrecision y, QPrecision z, QPrecision r_cutoff)
{
   QPrecision r2=(x*x+y*y+z*z);
   QPrecision r=SQRT(r2);

   VB[0]=0.0;	// reserved for future use, not used at present

   if (r>r_cutoff) {
      VB[1]=VB[2]=VB[3]=0.0;
      return;
   }
   if (r<RADIUS) {
      VB[1]=VB[3]=0.0;
      VB[2]=2.0*V0/3.0;
   } else {
      QPrecision coef=V0*RADIUS3/r2/r;
      VB[1]=coef*x*y/r2;
      VB[2]=coef*(y*y/r2-1.0/3.0);
      VB[3]=coef*z*y/r2;
   }
}
