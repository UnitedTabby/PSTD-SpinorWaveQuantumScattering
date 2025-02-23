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

#ifndef __QUTILITY__
#define __QUTILITY__

#include <fstream>
#include <string>
#include <numeric>
#include <stdlib.h> 
#include <asm-generic/errno-base.h>
#include <string.h>
#include "mkl.h"
#include "run_environment.h"

///////////////////////////////////////////////////////////////////////////////////////
// Initialize a vector
///////////////////////////////////////////////////////////////////////////////////////
template<class T> int Init_Vector(T *&X, long x0, long x1)
{
   long nx = x1 - x0 + 1;
   int val;

   val=posix_memalign((void **) &X, CACHE_LINE, nx*sizeof(T));
   if (val) return val;
   
   memset((void *) X, 0, nx*sizeof(T));
   X -= x0;

   return val;
};

///////////////////////////////////////////////////////////////////////////////////////
// Clear a vector
///////////////////////////////////////////////////////////////////////////////////////
template<class T> void Free_Vector(T *&X, long x0, long x1)
{
   if (X + x0) free(X + x0);

   X=NULL;
};

///////////////////////////////////////////////////////////////////////////////////////
// Initialize a 2D matrix
///////////////////////////////////////////////////////////////////////////////////////
template<class T> int Init_Matrix_2D(T **&X, long x0, long x1, long y0, long &y1, long padReq)
{
   long pad=(!padReq)?CACHE_LINE:(std::lcm(CACHE_LINE, padReq));

   long nx = x1 - x0 + 1;
   long ny = (((y1 - y0 + 1)*sizeof(T) + pad - 1)/pad) * pad / sizeof(T);
   long i;
   int val;

   X=(T**) calloc(nx, sizeof(T *));
   if (!X) return ENOMEM;

   long  s = nx*ny; 
   T *p;
   val=posix_memalign((void **) &p, CACHE_LINE, s*sizeof(T));
   if (val) 
   {
      free(X);
      return val;
   }
   
   memset((void *) p, 0, s*sizeof(T));

   for (i = 0, p -= y0; i < nx; i++)
   {
      X[i] = p;
      p += ny;
   }

   y1 = y0 + ny - 1;
   X -= x0;
   
   return val;
};

///////////////////////////////////////////////////////////////////////////////////////
// Clear a 2D matrix 
///////////////////////////////////////////////////////////////////////////////////////
template<class T> void Free_Matrix_2D(T **&X, long x0, long x1, long y0, long y1)
{
   if (!(X[x0] + y0)) return;
   free(X[x0] + y0);

   if (!(X + x0)) return;
   free(X + x0);

   X=NULL;
};

///////////////////////////////////////////////////////////////////////////////////////
// Initialize a 3D matrix
///////////////////////////////////////////////////////////////////////////////////////
template<class T> int Init_Matrix_3D(T ***&X, long x0, long x1, long y0, long y1, 
      long z0, long &z1, long padReq)
{
   long pad=(!padReq)?CACHE_LINE:(std::lcm(CACHE_LINE, padReq));

   long i, j;
   long nx = x1 - x0 + 1;
   long ny = y1 - y0 + 1;
   long nz = (((z1 - z0 + 1)*sizeof(T) + pad - 1)/pad) * pad / sizeof(T);
   int val;

   X = (T***) calloc(nx, sizeof(T **));
   if (!X) return ENOMEM;

   T **Y;
   Y = (T**) calloc(nx*ny, sizeof(T *));
   if (!Y) {
      free(X);
      return ENOMEM;
   }

   T *Z;
   val = posix_memalign(&Z, CACHE_LINE, nx*ny*nz*sizeof(T));
   if (val) {
      free(Y);
      free(X);
      return val;
   }
   memset(Z, 0, nx*ny*nz*sizeof(T));

   for (i = 0, Y -= y0, Z -= z0; i < nx; i++, Y += ny)
   {
      X[i] = Y;

      for (j = 0; j < ny; j++)
      {
         X[i][j] = Z;
         Z += nz;
      }
   }
   z1 = z0 + nz - 1;
   X -= x0;

   return val;
};

///////////////////////////////////////////////////////////////////////////////////////
// Clear a 3D matrix 
///////////////////////////////////////////////////////////////////////////////////////
template<class T> void Free_Matrix_3D(T ***&X, long x0, long x1, long y0,
      long y1, long z0, long z1)
{
   if (!(X[x0][y0] + z0)) free(X[x0][y0] + z0);
   if (!(X[x0] + y0)) free(X[x0] + y0);
   if (!(X + x0)) free(X + x0);
   X=NULL;
};

///////////////////////////////////////////////////////////////////////////////////////
// Initialize an MKL Vector
///////////////////////////////////////////////////////////////////////////////////////
template<class T> int Init_MKL_Vector(T *&X, long x0, long x1)
{
   long nx = x1 - x0 + 1;

   X = (T *) mkl_calloc(nx, sizeof(T), CACHE_LINE);
   if (!X ) return ENOMEM;

   X -= x0;
   return 0;
};

///////////////////////////////////////////////////////////////////////////////////////
// Free an MKL Vector
///////////////////////////////////////////////////////////////////////////////////////
template<class T> void Free_MKL_Vector(T *&X, long x0, long x1)
{
   if (X + x0) mkl_free(X + x0);
   X=NULL;
};

///////////////////////////////////////////////////////////////////////////////////////
// Initialize an MKL 2D matrix, used only in OpenMP environment for
// 1D-single-thread-FFTs, where each row serves 1 thread.  The purpose
// is to avoid false sharing between different cores.  Thus, no padding to
// the fastest varying index is attempted.  Yet, each row is aligned.
///////////////////////////////////////////////////////////////////////////////////////
template<class T> int Init_MKL_Matrix_2D(T **&X, long x0, long x1, long y0, long y1)
{
   long nx = x1 - x0 + 1;
   long ny = y1 - y0 + 1;

   X = (T **) mkl_calloc(nx, sizeof(T *), CACHE_LINE);
   if (!X) return ENOMEM;

   X -= x0;

   for (long i=x0; i<=x1; ++i) { // allocate nx 1D arrays to avoid false sharing
      X[i] = (T*) mkl_calloc(ny, sizeof(T), CACHE_LINE);
      if (!X[i]) {
	 for (long ii=0; ii<i; ++ii)
	    free(X[ii] + y0);
	 return ENOMEM;
      }
      X[i] -= y0;
   }

   return 0;
};

///////////////////////////////////////////////////////////////////////////////////////
// Clear an MKL 2D matrix 
///////////////////////////////////////////////////////////////////////////////////////
template<class T> void Free_MKL_Matrix_2D(T **&X, long x0, long x1, long y0, long y1)
{
   for (long i=x0; i<=x1; ++i)
      if (X[i] + y0) mkl_free(X[i] + y0);
   if (X + x0) mkl_free(X + x0);

   X=NULL;
};

///////////////////////////////////////////////////////////////////////////////////////
// Initialize an aligned vector
///////////////////////////////////////////////////////////////////////////////////////
template<class T> int Init_Aligned_Vector(T *&X, long nx)
{
   int val=posix_memalign((void **) &X, CACHE_LINE, nx*sizeof(T));
   if (!val) memset((void *) X, 0, nx*sizeof(T));

   return val;
};

///////////////////////////////////////////////////////////////////////////////////////
//Clear an aligned vector
///////////////////////////////////////////////////////////////////////////////////////
template<class T> void Free_Aligned_Vector(T *&X, long nx)
{
   if (X) {
      free(X);
      X=NULL;
   }
};

///////////////////////////////////////////////////////////////////////////////////////
//Initialize an aligned 2D matrix
///////////////////////////////////////////////////////////////////////////////////////
template<class T> int Init_Aligned_Matrix_2D(T *&X, long nx, long &ny, long padReq)
{
   long pad=(!padReq)?CACHE_LINE:(std::lcm(CACHE_LINE, padReq));

   ny = ((ny*sizeof(T) + pad - 1)/pad) * pad / sizeof(T);

   int val=posix_memalign((void **) &X, CACHE_LINE, nx*ny*sizeof(T));
   if (!val) memset((void *) X, 0, nx*ny*sizeof(T));

   return val;
};

///////////////////////////////////////////////////////////////////////////////////////
//Clear an aligned 2D matrix 
///////////////////////////////////////////////////////////////////////////////////////
template<class T> void Free_Aligned_Matrix_2D(T *&X, long nx, long ny)
{
   if (X) {
      free(X);
      X=NULL;
   }
};

///////////////////////////////////////////////////////////////////////////////////////
//Initialize an aligned 3D matrix
///////////////////////////////////////////////////////////////////////////////////////
template<class T> int Init_Aligned_Matrix_3D(T *&X, long nx, long ny, long &nz, long padReq)
{
   long pad=(!padReq)?CACHE_LINE:(std::lcm(CACHE_LINE, padReq));

   nz = ((nz*sizeof(T) + pad - 1)/pad) * pad / sizeof(T);

   int val=posix_memalign((void**) &X, CACHE_LINE, nx*ny*nz*sizeof(T));
   if (!val) memset((void *) X, 0, nx*ny*nz*sizeof(T));
   
   return val;
};

///////////////////////////////////////////////////////////////////////////////////////
//Clear an aligned 3D matrix 
///////////////////////////////////////////////////////////////////////////////////////
template<class T> void Free_Aligned_Matrix_3D(T *&X, long nx, long ny, long nz)
{
   if (X) {
      free(X);
      X=NULL;
   }
};

///////////////////////////////////////////////////////////////////////////////////////
// Initialize a padded & aligned MKL 2D matrix
///////////////////////////////////////////////////////////////////////////////////////
template<class T> int Init_Aligned_MKL_Matrix_2D(T *&X, long nx, long &ny, long padReq)
{
   long mklpad=(!padReq)?CACHE_LINE:(std::lcm(CACHE_LINE, padReq));

   ny = ((ny*sizeof(T) + mklpad - 1)/mklpad) * mklpad / sizeof(T);

   X = (T *) mkl_calloc(nx*ny, sizeof(T), CACHE_LINE);
   if (!X) return ENOMEM;

   memset((void*) X, 0, nx*ny*sizeof(T));

   return 0;
};

///////////////////////////////////////////////////////////////////////////////////////
// Free a padded & aligned MKL 2D matrix
///////////////////////////////////////////////////////////////////////////////////////
template<class T> void Free_Aligned_MKL_Matrix_2D(T *&X, long nx, long ny)
{
   if (X) {
      mkl_free(X);
      X=NULL;
   }
};

QPrecision ksi(QPrecision x);
QPrecision dksi(QPrecision x);
QPrecision d2ksi(QPrecision x);
void Derivative_K_Coeff_v2(QPrecision *k1, QPrecision *k2, QPrecision dD, long nD);
QPrecision weight(QPrecision x);
bool getaline(std::ifstream& ifs, std::string& aline);

#endif
