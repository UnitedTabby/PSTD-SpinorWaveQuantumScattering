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

#ifndef __RUN_ENVIRONMENT__
#define __RUN_ENVIRONMENT__

#include <cmath>

#define CACHE_LINE		64

#ifdef __USE_FLOAT__
  #define QPrecision		float
  #define DFTIPrecision		DFTI_SINGLE
  #define MKLComplex		MKL_Complex8
  #define MPIInt		MPI_INT
  #define MPILong		MPI_LONG
  #define MPIQPrecision		MPI_FLOAT
  #define SIN			sinf
  #define COS			cosf
  #define TAN			tanf
  #define LOG			logf
  #define EXP			expf
  #define SQRT			sqrtf
  #define FABS			fabsf
  #define TINY			1.0e-6f
  #define PI			3.141592653589793238462643383279502884197f
  #define TWOPI			6.283185307179586476925286766559005768394f
  #define FOURPI		12.56637061435917295385057353311801153679f
  #define SIXPI			18.84955592153875943077586029967701730518f
  #define MICROEV2NMBAR		4.552059563f

#else
  #define QPrecision		double
  #define DFTIPrecision		DFTI_DOUBLE
  #define MKLComplex		MKL_Complex16
  #define MPIInt		MPI_INT
  #define MPILong		MPI_LONG
  #define MPIQPrecision		MPI_DOUBLE
  #define SIN			sin
  #define COS			cos
  #define TAN			tan
  #define LOG			log
  #define EXP			exp
  #define SQRT			sqrt
  #define FABS			fabs
  #define TINY			1.0e-10
  #define PI			3.141592653589793238462643383279502884197
  #define TWOPI			6.283185307179586476925286766559005768394
  #define FOURPI		12.56637061435917295385057353311801153679
  #define SIXPI			18.84955592153875943077586029967701730518
  #define MICROEV2NMBAR		4.552059563
#endif

#define MAX_CHAR_PER_LINE       256

void Potential(QPrecision *V, QPrecision x, QPrecision y, QPrecision z, QPrecision r_cutoff);

#endif
