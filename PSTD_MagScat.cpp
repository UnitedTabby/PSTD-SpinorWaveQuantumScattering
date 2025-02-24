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

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <cmath>
#include <omp.h>
#include "PSTD_MagScat.h"
#include "utility.h"

// the cofficients for the finite difference in
// calculating the surface normal derivative of
// the scattering wavefunction on the virtual
// surface
#define FD1	4.0/5.0
#define FD2	-1.0/5.0
#define FD3	4.0/105.0
#define FD4	-1.0/280.0

using namespace std;

CPSTD_MagScat::CPSTD_MagScat()
{
   m_Psi0=NULL;
   m_Psi=NULL;

   m_g1=NULL;
   m_g2=NULL;
   m_g3=NULL;

   m_Inc_x1=NULL;
   m_Inc_x2=NULL;
   m_Inc_y1=NULL;
   m_Inc_y2=NULL;
   m_Inc_z1=NULL;
   m_Inc_z2=NULL;

   m_xTypeExch=0;
   m_yTypeExch=0;
   m_zTypeExch=0;

   m_descx=NULL;
   m_descy=NULL;
   m_descz=NULL;
   m_vFFT=NULL;
   m_k1x=NULL;
   m_k1y=NULL;
   m_k1z=NULL;
   m_k2x=NULL;
   m_k2y=NULL;
   m_k2z=NULL;

   m_SurfPsi_x1=NULL;
   m_SurfPsi_x2=NULL;
   m_SurfPsi_y1=NULL;
   m_SurfPsi_y2=NULL;
   m_SurfPsi_z1=NULL;
   m_SurfPsi_z2=NULL;
   m_SurfDPsi_PS_x1=NULL;
   m_SurfDPsi_PS_x2=NULL;
   m_SurfDPsi_PS_y1=NULL;
   m_SurfDPsi_PS_y2=NULL;
   m_SurfDPsi_PS_z1=NULL;
   m_SurfDPsi_PS_z2=NULL;
   m_SurfDPsi_FD_x1=NULL;
   m_SurfDPsi_FD_x2=NULL;
   m_SurfDPsi_FD_y1=NULL;
   m_SurfDPsi_FD_y2=NULL;
   m_SurfDPsi_FD_z1=NULL;
   m_SurfDPsi_FD_z2=NULL;

   m_VB_i=NULL;

   m_nThread=__OMP_NUM_THREADS__;	// number of threads will be set during make
}

void CPSTD_MagScat::Init(int nx_proc, int ny_proc, int nz_proc,
      QPrecision dtau, QPrecision dx, QPrecision dy, QPrecision dz,
      long nx_fft, long ny_fft, long nz_fft, long n_abc, long n_sca, long n_ksi, long n_halo,
      QPrecision theta, QPrecision phi, QPrecision E0, QPrecision *sp0,
      QPrecision CAP_U0, QPrecision CAP_alpha,
      void (*func) (QPrecision *, QPrecision, QPrecision, QPrecision, QPrecision),
      MPI_Comm comm)
{
   SetupCartComm(nx_proc, ny_proc, nz_proc, comm);
   m_dtau=dtau;
   m_dx=dx;
   m_dy=dy;
   m_dz=dz;
   m_nXFFT=nx_fft;
   m_nYFFT=ny_fft;
   m_nZFFT=nz_fft;
   m_nABC=n_abc;
   m_nSca=n_sca;
   m_nKsi=n_ksi;
   m_nHalo=n_halo;
   m_theta=theta;
   m_phi=phi;
   m_E0=E0;
   m_sp0[0]=sp0[0]; m_sp0[1]=sp0[1]; m_sp0[2]=sp0[2]; m_sp0[3]=sp0[3];
   m_CAP_U0=CAP_U0;
   m_CAP_alpha=CAP_alpha;
   m_lambdabar0=MICROEV2NMBAR/SQRT(m_E0);
   m_alpha_2dtau_i=2.0*SIN(m_dtau);
   DomainDecomposition();
   InitMPIExchange();

   m_OrigX=(QPrecision)(m_nXGlb-1)/2.0;
   m_OrigY=(QPrecision)(m_nYGlb-1)/2.0;
   m_OrigZ=(QPrecision)(m_nZGlb-1)/2.0;

   for (long i=0; i<NWEIGHT; ++i)
      m_weight[i]=weight((QPrecision) (i+1) / (QPrecision) (NWEIGHT+1));

   InitFFT();
   InitABC();
   InitTS();
   InitPsi();
   InitV(func);
   InitVirtualSurfaces();

   // when __INIT_STATUS__ is set on the make command line, show the 
   // initialized configuration to the standout
#ifdef __INIT_STATUS__
   WriteInitStatusToSTDOUT();
#endif
}

// retrieve the dimensions of nodes (processes), rank coordinate of the local node,
// and the rank coordinates of the neighboring nodes (back/front/left/right/bottom/top neighbors)
void CPSTD_MagScat::SetupCartComm(int nx_proc, int ny_proc, int nz_proc, MPI_Comm comm)
{
   int nproc;
   MPI_Comm_size(comm, &nproc);
   if (nproc!=nx_proc*ny_proc*nz_proc) {
      stringstream sstr;
      sstr << "Error: # of processes in data and # of running processes mismatch -- ("
	 << nx_proc << "," << ny_proc << "," << nz_proc << ") vs. " << nproc;
      throw runtime_error(sstr.str());
   }
   int dims[3], periods[3], coords[3];
   dims[0]=nx_proc; dims[1]=ny_proc; dims[2]=nz_proc;
   periods[0]=periods[1]=periods[2]=0;
   MPI_Cart_create(comm, 3, dims, periods, 1, &m_cartcomm);
   MPI_Cart_get(m_cartcomm, 3, dims, periods, coords);
   m_nXProcs=dims[0];
   m_nYProcs=dims[1];
   m_nZProcs=dims[2];
   m_xRank=coords[0];
   m_yRank=coords[1];
   m_zRank=coords[2];
   MPI_Comm_rank(m_cartcomm, &m_Rank); 
   if (!m_Rank) 
      cout << "Process topology (x, y, z): " << \
	 m_nXProcs << ", " << m_nYProcs << ", " << m_nZProcs << endl;
   MPI_Cart_shift(m_cartcomm, 0, 1, &m_xaRank, &m_xbRank);
   MPI_Cart_shift(m_cartcomm, 1, 1, &m_yaRank, &m_ybRank);
   MPI_Cart_shift(m_cartcomm, 2, 1, &m_zaRank, &m_zbRank);
}

void CPSTD_MagScat::DomainDecomposition()
{
   // overlapping domain decomposition for the x dimension:
   // Because the length of FFT needs to factorize into products of special radices for
   // performance reasons, we set nx_fft as input and derive the total number of
   // grids accordingly.  Refer to length_advisor.c for optimal choice of nx_fft.
   // the 1st & last processes have overlapping at only one side
   m_nXGlb=m_nXProcs*(m_nXFFT-2*m_nHalo)+2*m_nHalo;
   if (m_nXProcs==1) { // only one process in the x direction
      m_x1Core=m_x1FFT=0;
      m_x2Core=m_x2FFT=m_nXFFT-1;
   } else if (m_xRank==0) { // no back overlapping for the 1st x-rank
      m_x1Core=m_x1FFT=0;
      m_x2FFT=m_x1FFT+m_nXFFT-1;
      m_x2Core=m_x2FFT-m_nHalo;
   } else {
      m_x1Core=m_xRank*(m_nXFFT-2*m_nHalo)+m_nHalo;
      m_x1FFT=m_x1Core-m_nHalo;
      m_x2FFT=m_x1FFT+m_nXFFT-1;
      m_x2Core=(m_xRank==m_nXProcs-1)?m_x2FFT:(m_x2FFT-m_nHalo); // no front overlapping for the last x-rank
   }
   m_nxCore=m_x2Core-m_x1Core+1;
   m_i1Core=m_x1Core-m_x1FFT;
   m_i2Core=m_x2Core-m_x1FFT;

   // overlapping domain decomposition for the y dimension:
   m_nYGlb=m_nYProcs*(m_nYFFT-2*m_nHalo)+2*m_nHalo;
   if (m_nYProcs==1) { // only one process in the y direction
      m_y1Core=m_y1FFT=0;
      m_y2Core=m_y2FFT=m_nYFFT-1;
   } else if (m_yRank==0) { // no left overlapping for the 1st y-rank
      m_y1Core=m_y1FFT=0;
      m_y2FFT=m_y1FFT+m_nYFFT-1;
      m_y2Core=m_y2FFT-m_nHalo;
   } else {
      m_y1Core=m_yRank*(m_nYFFT-2*m_nHalo)+m_nHalo;
      m_y1FFT=m_y1Core-m_nHalo;
      m_y2FFT=m_y1FFT+m_nYFFT-1;
      m_y2Core=(m_yRank==m_nYProcs-1)?m_y2FFT:(m_y2FFT-m_nHalo); // no right overlapping for the last y-rank
   }
   m_nyCore=m_y2Core-m_y1Core+1;
   m_j1Core=m_y1Core-m_y1FFT;
   m_j2Core=m_y2Core-m_y1FFT;

   // overlapping domain decomposition for the z dimension:
   m_nZGlb=m_nZProcs*(m_nZFFT-2*m_nHalo)+2*m_nHalo;
   if (m_nZProcs==1) { // only one process in the z direction
      m_z1Core=m_z1FFT=0;
      m_z2Core=m_z2FFT=m_nZFFT-1;
   } else if (m_zRank==0) { // no bottom overlapping for the 1st z-rank
      m_z1Core=m_z1FFT=0;
      m_z2FFT=m_z1FFT+m_nZFFT-1;
      m_z2Core=m_z2FFT-m_nHalo;
   } else {
      m_z1Core=m_zRank*(m_nZFFT-2*m_nHalo)+m_nHalo;
      m_z1FFT=m_z1Core-m_nHalo;
      m_z2FFT=m_z1FFT+m_nZFFT-1;
      m_z2Core=(m_zRank==m_nZProcs-1)?m_z2FFT:(m_z2FFT-m_nHalo); // no top overlapping for the last z-rank
   }
   m_nzCore=m_z2Core-m_z1Core+1;
   m_k1Core=m_z1Core-m_z1FFT;
   m_k2Core=m_z2Core-m_z1FFT;

   // Find the absorbing boundaries
   m_X1ABC=m_nABC-1;
   m_X2ABC=m_nXGlb-m_nABC;
   m_Y1ABC=m_nABC-1;
   m_Y2ABC=m_nYGlb-m_nABC;
   m_Z1ABC=m_nABC-1;
   m_Z2ABC=m_nZGlb-m_nABC;

   // Find the boundaries of the transition zone, i.e. the TF-SF interface
   m_X1aKsi=m_X1ABC+m_nSca+1;
   m_X1bKsi=m_X1aKsi+m_nKsi-1;
   m_X2bKsi=m_X2ABC-m_nSca-1;
   m_X2aKsi=m_X2bKsi-m_nKsi+1;
   m_Y1aKsi=m_Y1ABC+m_nSca+1;
   m_Y1bKsi=m_Y1aKsi+m_nKsi-1;
   m_Y2bKsi=m_Y2ABC-m_nSca-1;
   m_Y2aKsi=m_Y2bKsi-m_nKsi+1;
   m_Z1aKsi=m_Z1ABC+m_nSca+1;
   m_Z1bKsi=m_Z1aKsi+m_nKsi-1;
   m_Z2bKsi=m_Z2ABC-m_nSca-1;
   m_Z2aKsi=m_Z2bKsi-m_nKsi+1;

   // allocate Psi0, Psi
   m_nx=m_nXFFT;
   m_ny=m_nYFFT;
   m_nz=m_nZFFT*4;
   int val=Init_Aligned_Matrix_3D<QPrecision>(m_Psi0, m_nx, m_ny, m_nz, sizeof(QPrecision)*4);
   if (val) HandleError(val, "cannot allocate m_Psi0");

   val=Init_Aligned_Matrix_3D<QPrecision>(m_Psi, m_nx, m_ny, m_nz, sizeof(QPrecision)*4);
   if (val) HandleError(val, "cannot allocate m_Psi");

   m_nyz=m_ny*m_nz;

   // allocate memory for the potential
   if (m_x1Core>=m_X2aKsi || m_x2Core<=m_X1bKsi || m_y1Core>=m_Y2aKsi 
	 || m_y2Core<=m_Y1bKsi || m_z1Core>=m_Z2aKsi || m_z2Core<=m_Z1bKsi)
      m_flgV=false;
   else {
      m_flgV=true;
      m_x1V=(m_x1Core>m_X1bKsi)?m_x1Core:(m_X1bKsi+1);
      m_x2V=(m_x2Core<m_X2aKsi)?m_x2Core:(m_X2aKsi-1);
      m_y1V=(m_y1Core>m_Y1bKsi)?m_y1Core:(m_Y1bKsi+1);
      m_y2V=(m_y2Core<m_Y2aKsi)?m_y2Core:(m_Y2aKsi-1);
      m_z1V=(m_z1Core>m_Z1bKsi)?m_z1Core:(m_Z1bKsi+1);
      m_z2V=(m_z2Core<m_Z2aKsi)?m_z2Core:(m_Z2aKsi-1);
      m_nxV=m_x2V-m_x1V+1;
      m_nyV=m_y2V-m_y1V+1;
      m_nzV=(m_z2V-m_z1V+1)*4;

      val=Init_Aligned_Matrix_3D<QPrecision>(m_VB_i, m_nxV, m_nyV, m_nzV, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_VB_i");

      m_nyzV=m_nyV*m_nzV;
   }

   // Find the locations of the virtual surfaces
   m_X1Vrtl=m_X1ABC+(m_nSca+1)/2;  // not a floating-point value division; doesn't matter wheter m_nSca is even or odd
   m_X2Vrtl=m_X2ABC-(m_nSca+1)/2;
   m_Y1Vrtl=m_Y1ABC+(m_nSca+1)/2;
   m_Y2Vrtl=m_Y2ABC-(m_nSca+1)/2;
   m_Z1Vrtl=m_Z1ABC+(m_nSca+1)/2;
   m_Z2Vrtl=m_Z2ABC-(m_nSca+1)/2;
}

// the exchanged date are from the core; the core is offsetted in the local
// date due to the halo.  we must identify the date structure for exchanging
// beforehand.
void CPSTD_MagScat::InitMPIExchange()
{
   // exchange type along x direction for the front and back surfaces
   long nxCore, nyCore, nzCore;
   int *blocklen, *indx;
   long count;
   long halo_1=m_nHalo-1;	// the weight factor for the last one is always 0, so no need to exchange it.
   int val;

   // exchange type along x direction for the back and front surfaces
   if (m_nXProcs>1) {
      count=halo_1*m_nyCore;
      val=Init_Aligned_Vector<int>(blocklen, count);
      if (val) HandleError(val, "cannot allocate blocklen for m_xTypeExch");

      val=Init_Aligned_Vector<int>(indx, count);
      if (val) HandleError(val, "cannot allocate indx for m_xTypeExch");

#pragma omp parallel for num_threads(m_nThread)
      for (long i=0; i<halo_1; ++i)
#pragma omp simd aligned(blocklen:CACHE_LINE) aligned(indx:CACHE_LINE)
	 for (long j=0; j<m_nyCore; ++j) {
	    blocklen[i*m_nyCore+j]=m_nzCore*4;
	    indx[i*m_nyCore+j]=i*m_nyz+j*m_nz;
	 }

      MPI_Type_indexed(count,blocklen,indx,MPIQPrecision,&m_xTypeExch);
      MPI_Type_commit(&m_xTypeExch);
      Free_Aligned_Vector<int>(blocklen, count);
      Free_Aligned_Vector<int>(indx, count);
   }

   // exchange type along y direction for the left and right surfaces
   if (m_nYProcs>1) {
      count=m_nxCore*halo_1;
      val=Init_Aligned_Vector<int>(blocklen, count);
      if (val) HandleError(val, "cannot allocate blocklen for m_yTypeExch");

      val=Init_Aligned_Vector<int>(indx, count);
      if (val) HandleError(val, "cannot allocate indx for m_yTypeExch");

#pragma omp parallel for num_threads(m_nThread)
      for (long i=0; i<m_nxCore; ++i)
#pragma omp simd aligned(blocklen:CACHE_LINE) aligned(indx:CACHE_LINE)
	 for (long j=0; j<halo_1; ++j) {
	    blocklen[i*halo_1+j]=m_nzCore*4;
	    indx[i*halo_1+j]=i*m_nyz+j*m_nz;
	 }

      MPI_Type_indexed(count,blocklen,indx,MPIQPrecision,&m_yTypeExch);
      MPI_Type_commit(&m_yTypeExch);
      Free_Aligned_Vector<int>(blocklen, count);
      Free_Aligned_Vector<int>(indx, count);
   }

   // exchange type along z direction for the bottom and top surfaces
   if (m_nZProcs>1) {
      count=m_nxCore*m_nyCore;
      val=Init_Aligned_Vector<int>(blocklen, count);
      if (val) HandleError(val, "cannot allocate blocklen for m_zTypeExch");

      val=Init_Aligned_Vector<int>(indx, count);
      if (val) HandleError(val, "cannot allocate indx for m_zTypeExch");

#pragma omp parallel for num_threads(m_nThread)
      for (long i=0; i<m_nxCore; ++i)
#pragma omp simd aligned(blocklen:CACHE_LINE) aligned(indx:CACHE_LINE)
	 for (long j=0; j<m_nyCore; ++j) {
	    blocklen[i*m_nyCore+j]=halo_1*4;
	    indx[i*m_nyCore+j]=i*m_nyz+j*m_nz;
	 }

      MPI_Type_indexed(count,blocklen,indx,MPIQPrecision,&m_zTypeExch);
      MPI_Type_commit(&m_zTypeExch);
      Free_Aligned_Vector<int>(blocklen, count);
      Free_Aligned_Vector<int>(indx, count);
   }

   m_xaSendOffset=m_i1Core*m_nyz+m_j1Core*m_nz+m_k1Core*4;
   m_xaRecvOffset=m_nyz+m_j1Core*m_nz+m_k1Core*4;
   m_xbSendOffset=(m_i2Core-(m_nHalo-2))*m_nyz+m_j1Core*m_nz+m_k1Core*4;
   m_xbRecvOffset=(m_i2Core+1)*m_nyz+m_j1Core*m_nz+m_k1Core*4;

   m_yaSendOffset=m_i1Core*m_nyz+m_j1Core*m_nz+m_k1Core*4;
   m_yaRecvOffset=m_i1Core*m_nyz+m_nz+m_k1Core*4;
   m_ybSendOffset=m_i1Core*m_nyz+(m_j2Core-(m_nHalo-2))*m_nz+m_k1Core*4;
   m_ybRecvOffset=m_i1Core*m_nyz+(m_j2Core+1)*m_nz+m_k1Core*4;

   m_zaSendOffset=m_i1Core*m_nyz+m_j1Core*m_nz+m_k1Core*4;
   m_zaRecvOffset=m_i1Core*m_nyz+m_j1Core*m_nz+4;
   m_zbSendOffset=m_i1Core*m_nyz+m_j1Core*m_nz+(m_k2Core-(m_nHalo-2))*4;
   m_zbRecvOffset=m_i1Core*m_nyz+m_j1Core*m_nz+(m_k2Core+1)*4;
}

// Intel MKL FFT libraries requires an initialization
void CPSTD_MagScat::InitFFT()
{
   m_descx=(DFTI_DESCRIPTOR_HANDLE *) calloc(m_nThread, sizeof(DFTI_DESCRIPTOR_HANDLE));
   if (!m_descx) HandleError(ENOMEM, "cannot allocate m_descx");

   for (int i=0; i<m_nThread; ++i) {
      DftiCreateDescriptor(&m_descx[i], DFTIPrecision, DFTI_COMPLEX, 1, m_nXFFT);
      DftiSetValue(m_descx[i],DFTI_FORWARD_SCALE,1.0);
      DftiSetValue(m_descx[i],DFTI_BACKWARD_SCALE,1.0/(QPrecision) m_nXFFT);
      DftiCommitDescriptor(m_descx[i]);
   }

   m_descy=(DFTI_DESCRIPTOR_HANDLE *) calloc(m_nThread, sizeof(DFTI_DESCRIPTOR_HANDLE));
   if (!m_descy) HandleError(ENOMEM, "cannot allocate m_descy");

   for (int j=0; j<m_nThread; ++j) {
      DftiCreateDescriptor(&m_descy[j], DFTIPrecision, DFTI_COMPLEX, 1, m_nYFFT);
      DftiSetValue(m_descy[j],DFTI_FORWARD_SCALE,1.0);
      DftiSetValue(m_descy[j],DFTI_BACKWARD_SCALE,1.0/(QPrecision) m_nYFFT);
      DftiCommitDescriptor(m_descy[j]);
   }

   m_descz=(DFTI_DESCRIPTOR_HANDLE *) calloc(m_nThread, sizeof(DFTI_DESCRIPTOR_HANDLE));
   if (!m_descz) HandleError(ENOMEM, "cannot allocate m_descz");

   for (int k=0; k<m_nThread; ++k) {
      DftiCreateDescriptor(&m_descz[k], DFTIPrecision, DFTI_COMPLEX, 1, m_nZFFT);
      DftiSetValue(m_descz[k],DFTI_FORWARD_SCALE,1.0);
      DftiSetValue(m_descz[k],DFTI_BACKWARD_SCALE,1.0/(QPrecision) m_nZFFT);
      DftiCommitDescriptor(m_descz[k]);
   }

   // memory allocation using mkl_calloc.  We allocate m_nThread vectors, instead of
   // one single (m_nThread x vlen) matrix, in order to avoid openmp false sharing.
   int val=Init_MKL_Matrix_2D<MKLComplex>(m_vFFT, 0, m_nThread-1, 0,
	 max(max(m_nXFFT, m_nYFFT), m_nZFFT)-1);
   if (val) HandleError(val, "cannot allocate m_vFFT");

   // for FFT derivative operations
   // the 1st derivative is offset by half a grid; zero offset for 2nd derivative 
   val=Init_Aligned_Vector<QPrecision>(m_k1x,m_nXFFT*2);
   if (val) HandleError(val, "cannot allocate m_k1x");

   val=Init_Aligned_Vector<QPrecision>(m_k1y,m_nYFFT*2);
   if (val) HandleError(val, "cannot allocate m_k1y");

   val=Init_Aligned_Vector<QPrecision>(m_k1z,m_nZFFT*2);
   if (val) HandleError(val, "cannot allocate m_k1z");

   val=Init_Aligned_Vector<QPrecision>(m_k2x,m_nXFFT);
   if (val) HandleError(val, "cannot allocate m_k2x");

   val=Init_Aligned_Vector<QPrecision>(m_k2y,m_nYFFT);
   if (val) HandleError(val, "cannot allocate m_k2y");

   val=Init_Aligned_Vector<QPrecision>(m_k2z,m_nZFFT);
   if (val) HandleError(val, "cannot allocate m_k2z");

   Derivative_K_Coeff_v2(m_k1x, m_k2x, m_dx, m_nXFFT);
   Derivative_K_Coeff_v2(m_k1y, m_k2y, m_dy, m_nYFFT);
   Derivative_K_Coeff_v2(m_k1z, m_k2z, m_dz, m_nZFFT);
}

// initiate the absorbing boundary condition
void CPSTD_MagScat::InitABC()
{
   QPrecision *g;

   // x direction
   int val=Init_Aligned_Vector<QPrecision>(m_g1,m_nxCore);
   if (val) HandleError(val, "cannot allocate m_g1");

   g=m_g1;
#pragma omp parallel for num_threads(m_nThread)
   for (long i=m_x1Core; i<=m_x2Core; ++i) {
      QPrecision tmp;
      if (i<(m_nXGlb>>1))
	 tmp=cosh(m_CAP_alpha*(double) i);
      else
	 tmp=cosh(m_CAP_alpha*(double) (m_nXGlb-1-i));
      *(g+i-m_x1Core)=1.0-m_dtau*m_CAP_U0/(tmp*tmp);
   }

   // y direction
   val=Init_Aligned_Vector<QPrecision>(m_g2,m_nyCore);
   if (val) HandleError(val, "cannot allocate m_g2");

   g=m_g2;
#pragma omp parallel for num_threads(m_nThread)
   for (long j=m_y1Core; j<=m_y2Core; ++j) {
      QPrecision tmp;
      if (j<(m_nYGlb>>1))
	 tmp=cosh(m_CAP_alpha*(double) j);
      else
	 tmp=cosh(m_CAP_alpha*(double) (m_nYGlb-1-j));
      *(g+j-m_y1Core)=1.0-m_dtau*m_CAP_U0/(tmp*tmp);
   }

   // z direction
   val=Init_Aligned_Vector<QPrecision>(m_g3,m_nzCore);
   if (val) HandleError(val, "cannot allocate m_g3");

   g=m_g3;
#pragma omp parallel for num_threads(m_nThread)
   for (long k=m_z1Core; k<=m_z2Core; ++k) {
      QPrecision tmp;
      if (k<(m_nZGlb>>1))
	 tmp=cosh(m_CAP_alpha*(double) k);
      else
	 tmp=cosh(m_CAP_alpha*(double) (m_nZGlb-1-k));
      *(g+k-m_z1Core)=1.0-m_dtau*m_CAP_U0/(tmp*tmp);
   }
}

// intialize the Total-field/scattered-field, i.e. the incident wave
// source condition.  This code is only valid for cw plane wave incidence,
// and the $\exp(ik\vec{r})$ wave form is hard coded into the variables.
void CPSTD_MagScat::InitTS()
{
   // Find the injection point of the incident wave
   QPrecision cos_theta=COS(m_theta);
   QPrecision sin_theta=SIN(m_theta);
   QPrecision cos_phi=COS(m_phi);
   QPrecision sin_phi=SIN(m_phi);
   long &i0=m_XInj, &j0=m_YInj, &k0=m_ZInj;
   if (cos_theta>=0.0) {
      if (sin_phi>=0.0) {
	 if (cos_phi>=0.0) {
	    i0=m_X1aKsi; j0=m_Y1aKsi; k0=m_Z1aKsi;
	 } else {
	    i0=m_X2bKsi; j0=m_Y1aKsi; k0=m_Z1aKsi;
	 }
      } else {
	 if (cos_phi>=0.0) {
	    i0=m_X1aKsi; j0=m_Y2bKsi; k0=m_Z1aKsi;
	 } else {
	    i0=m_X2bKsi; j0=m_Y2bKsi; k0=m_Z1aKsi;
	 }
      }
   } else {
      if (sin_phi>=0.0) {
	 if (cos_phi>=0.0) {
	    i0=m_X1aKsi; j0=m_Y1aKsi; k0=m_Z2bKsi;
	 } else {
	    i0=m_X2bKsi; j0=m_Y1aKsi; k0=m_Z2bKsi;
	 }
      } else {
	 if (cos_phi>=0.0) {
	    i0=m_X1aKsi; j0=m_Y2bKsi; k0=m_Z2bKsi;
	 } else {
	    i0=m_X2bKsi; j0=m_Y2bKsi; k0=m_Z2bKsi;
	 }
      }
   }

   int val;
   QPrecision dx=m_dx*sin_theta*cos_phi;
   QPrecision dy=m_dy*sin_theta*sin_phi;
   QPrecision dz=m_dz*cos_theta;
   QPrecision dtc_r=COS(m_dtau);
   QPrecision dtc_i=-SIN(m_dtau);

   // Initialize the transition zone
   m_flgx1Ksi=m_flgx2Ksi=m_flgy1Ksi=m_flgy2Ksi=m_flgz1Ksi=m_flgz2Ksi=false;

   if (m_x2Core<m_X1aKsi || m_x1Core>m_X2bKsi || m_y2Core<m_Y1aKsi || m_y1Core>m_Y2bKsi
	 || m_z2Core<m_Z1aKsi || m_z1Core>m_Z2bKsi || m_x1Core>m_X1bKsi && m_x2Core<m_X2aKsi
	 && m_y1Core>m_Y1bKsi && m_y2Core<m_Y2aKsi && m_z1Core>m_Z1bKsi && m_z2Core<m_Z2aKsi)
      return;  // non-crossing situations excluded

   if (m_x1Core<=m_X1bKsi) m_flgx1Ksi=true;
   if (m_x2Core>=m_X2aKsi) m_flgx2Ksi=true;
   if (m_y1Core<=m_Y1bKsi) m_flgy1Ksi=true;
   if (m_y2Core>=m_Y2aKsi) m_flgy2Ksi=true;
   if (m_z1Core<=m_Z1bKsi) m_flgz1Ksi=true;
   if (m_z2Core>=m_Z2aKsi) m_flgz2Ksi=true;

   if (m_flgx1Ksi) {
      m_slab_x1.xa=(m_x1Core>m_X1aKsi)?m_x1Core:m_X1aKsi;
      m_slab_x1.xb=(m_x2Core<m_X1bKsi)?m_x2Core:m_X1bKsi;
      m_slab_x1.ya=(m_y1Core>m_Y1aKsi)?m_y1Core:m_Y1aKsi;
      m_slab_x1.yb=(m_y2Core<m_Y2bKsi)?m_y2Core:m_Y2bKsi;
      m_slab_x1.za=(m_z1Core>m_Z1aKsi)?m_z1Core:m_Z1aKsi;
      m_slab_x1.zb=(m_z2Core<m_Z2bKsi)?m_z2Core:m_Z2bKsi;

      m_slab_x1.nx=m_slab_x1.xb-m_slab_x1.xa+1;	// only used for allocating/indexing/freeing m_Inc_x1
      m_slab_x1.ny=m_slab_x1.yb-m_slab_x1.ya+1;	//
      m_slab_x1.nz=(m_slab_x1.zb-m_slab_x1.za+1)*4;	// m_slab_x1.nz subjected to change when allocating m_Inc_x1
      val=Init_Aligned_Matrix_3D<QPrecision>(m_Inc_x1,
	    m_slab_x1.nx, m_slab_x1.ny, m_slab_x1.nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_Inc_x1");
   }

   if (m_flgx2Ksi) {
      m_slab_x2.xa=(m_x1Core>m_X2aKsi)?m_x1Core:m_X2aKsi;
      m_slab_x2.xb=(m_x2Core<m_X2bKsi)?m_x2Core:m_X2bKsi;
      m_slab_x2.ya=(m_y1Core>m_Y1aKsi)?m_y1Core:m_Y1aKsi;
      m_slab_x2.yb=(m_y2Core<m_Y2bKsi)?m_y2Core:m_Y2bKsi;
      m_slab_x2.za=(m_z1Core>m_Z1aKsi)?m_z1Core:m_Z1aKsi;
      m_slab_x2.zb=(m_z2Core<m_Z2bKsi)?m_z2Core:m_Z2bKsi;

      m_slab_x2.nx=m_slab_x2.xb-m_slab_x2.xa+1;	// only used for allocating/indexing/freeing m_Inc_x2
      m_slab_x2.ny=m_slab_x2.yb-m_slab_x2.ya+1;	//
      m_slab_x2.nz=(m_slab_x2.zb-m_slab_x2.za+1)*4;	// m_slab_x2.nz subjected to change when allocating m_Inc_x2
      val=Init_Aligned_Matrix_3D<QPrecision>(m_Inc_x2,
	    m_slab_x2.nx, m_slab_x2.ny, m_slab_x2.nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_Inc_x2");
   }

   if (m_flgy1Ksi) {
      m_slab_y1.xa=(m_x1Core>m_X1aKsi)?m_x1Core:m_X1aKsi;
      m_slab_y1.xb=(m_x2Core<m_X2bKsi)?m_x2Core:m_X2bKsi;
      m_slab_y1.ya=(m_y1Core>m_Y1aKsi)?m_y1Core:m_Y1aKsi;
      m_slab_y1.yb=(m_y2Core<m_Y1bKsi)?m_y2Core:m_Y1bKsi;
      m_slab_y1.za=(m_z1Core>m_Z1aKsi)?m_z1Core:m_Z1aKsi;
      m_slab_y1.zb=(m_z2Core<m_Z2bKsi)?m_z2Core:m_Z2bKsi;

      m_slab_y1.nx=m_slab_y1.xb-m_slab_y1.xa+1;	// only used for allocating/indexing/freeing m_Inc_y1
      m_slab_y1.ny=m_slab_y1.yb-m_slab_y1.ya+1; //
      m_slab_y1.nz=(m_slab_y1.zb-m_slab_y1.za+1)*4;	// m_slab_y1.nz subjected to change when allocating m_Inc_y1
      val=Init_Aligned_Matrix_3D<QPrecision>(m_Inc_y1,
	    m_slab_y1.nx, m_slab_y1.ny, m_slab_y1.nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_Inc_y1");
   }

   if (m_flgy2Ksi) {
      m_slab_y2.xa=(m_x1Core>m_X1aKsi)?m_x1Core:m_X1aKsi;
      m_slab_y2.xb=(m_x2Core<m_X2bKsi)?m_x2Core:m_X2bKsi;
      m_slab_y2.ya=(m_y1Core>m_Y2aKsi)?m_y1Core:m_Y2aKsi;
      m_slab_y2.yb=(m_y2Core<m_Y2bKsi)?m_y2Core:m_Y2bKsi;
      m_slab_y2.za=(m_z1Core>m_Z1aKsi)?m_z1Core:m_Z1aKsi;
      m_slab_y2.zb=(m_z2Core<m_Z2bKsi)?m_z2Core:m_Z2bKsi;

      m_slab_y2.nx=m_slab_y2.xb-m_slab_y2.xa+1;	// only used for allocating/indexing/freeing m_Inc_y2
      m_slab_y2.ny=m_slab_y2.yb-m_slab_y2.ya+1;	//
      m_slab_y2.nz=(m_slab_y2.zb-m_slab_y2.za+1)*4;	// m_slab_y2.nz subjected to change when allocating m_Inc_y2
      val=Init_Aligned_Matrix_3D<QPrecision>(m_Inc_y2,
	    m_slab_y2.nx, m_slab_y2.ny, m_slab_y2.nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_Inc_y2");
   }

   if (m_flgz1Ksi) {
      m_slab_z1.xa=(m_x1Core>m_X1aKsi)?m_x1Core:m_X1aKsi;
      m_slab_z1.xb=(m_x2Core<m_X2bKsi)?m_x2Core:m_X2bKsi;
      m_slab_z1.ya=(m_y1Core>m_Y1aKsi)?m_y1Core:m_Y1aKsi;
      m_slab_z1.yb=(m_y2Core<m_Y2bKsi)?m_y2Core:m_Y2bKsi;
      m_slab_z1.za=(m_z1Core>m_Z1aKsi)?m_z1Core:m_Z1aKsi;
      m_slab_z1.zb=(m_z2Core<m_Z1bKsi)?m_z2Core:m_Z1bKsi;

      m_slab_z1.nx=m_slab_z1.xb-m_slab_z1.xa+1;	// only used for allocating/indexing/freeing m_Inc_z1
      m_slab_z1.ny=m_slab_z1.yb-m_slab_z1.ya+1;	//
      m_slab_z1.nz=(m_slab_z1.zb-m_slab_z1.za+1)*4;	// m_slab_z1.nz subjected to change when allocating m_Inc_z1
      val=Init_Aligned_Matrix_3D<QPrecision>(m_Inc_z1,
	    m_slab_z1.nx, m_slab_z1.ny, m_slab_z1.nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_Inc_z1");
   }

   if (m_flgz2Ksi) {
      m_slab_z2.xa=(m_x1Core>m_X1aKsi)?m_x1Core:m_X1aKsi;
      m_slab_z2.xb=(m_x2Core<m_X2bKsi)?m_x2Core:m_X2bKsi;
      m_slab_z2.ya=(m_y1Core>m_Y1aKsi)?m_y1Core:m_Y1aKsi;
      m_slab_z2.yb=(m_y2Core<m_Y2bKsi)?m_y2Core:m_Y2bKsi;
      m_slab_z2.za=(m_z1Core>m_Z2aKsi)?m_z1Core:m_Z2aKsi;
      m_slab_z2.zb=(m_z2Core<m_Z2bKsi)?m_z2Core:m_Z2bKsi;

      m_slab_z2.nx=m_slab_z2.xb-m_slab_z2.xa+1;	// only used for allocating/indexing/freeing m_Inc_z2
      m_slab_z2.ny=m_slab_z2.yb-m_slab_z2.ya+1;	//
      m_slab_z2.nz=(m_slab_z2.zb-m_slab_z2.za+1)*4;	// m_slab_z2.nz subjected to change when allocating m_Inc_z2
      val=Init_Aligned_Matrix_3D<QPrecision>(m_Inc_z2,
	    m_slab_z2.nx, m_slab_z2.ny, m_slab_z2.nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_Inc_z2");
   }

   QPrecision n0t1=(QPrecision) (m_nKsi-1); // m_nKsi grids, (m_nKsi-1) segments
   if (m_flgx1Ksi) {
      QPrecision l=n0t1*m_dx;
      QPrecision coef=2.0*m_dtau/l;
      long nz=m_slab_x1.nz;
      long nyz=m_slab_x1.ny*nz;
#pragma omp parallel for num_threads(m_nThread)
      for (long j=m_slab_x1.ya; j<=m_slab_x1.yb; ++j) {
	 QPrecision rj=(j-j0)*dy;
	 QPrecision tmp1=ksi((QPrecision) (j-m_Y1aKsi)/n0t1)
	    *ksi((QPrecision) (m_Y2bKsi-j)/n0t1)*coef;
	 // thickness of slab is thin, not enough trip count for m_nThread,
	 // placed as inner loop, the i-loop won't be parallelized
	 for (long i=m_slab_x1.xa; i<=m_slab_x1.xb; ++i) {
	    QPrecision rij=rj+(i-i0)*dx;
	    QPrecision x=(QPrecision) (i-m_X1aKsi)/n0t1;
	    QPrecision tmp_r=tmp1*d2ksi(x)/l;
	    QPrecision tmp_i=tmp1*2.0*dksi(x)*sin_theta*cos_phi;
	    QPrecision *pInc=m_Inc_x1+(i-m_slab_x1.xa)*nyz+(j-m_slab_x1.ya)*nz;
#pragma omp simd aligned(pInc:CACHE_LINE)
	    for (long k=m_slab_x1.za; k<=m_slab_x1.zb; ++k) {
	       QPrecision r=rij+(k-k0)*dz;
	       QPrecision cosr=COS(r);
	       QPrecision sinr=SIN(r);
	       QPrecision tmp2=ksi((QPrecision) (k-m_Z1aKsi)/n0t1)
		  *ksi((QPrecision) (m_Z2bKsi-k)/n0t1);
	       QPrecision ttmp_r=tmp2*(tmp_r*cosr-tmp_i*sinr);
	       QPrecision ttmp_i=tmp2*(tmp_r*sinr+tmp_i*cosr);
	       *(pInc+(k-m_slab_x1.za)*4)=ttmp_r*m_sp0[0]-ttmp_i*m_sp0[1];
	       *(pInc+(k-m_slab_x1.za)*4+1)=ttmp_r*m_sp0[1]+ttmp_i*m_sp0[0];
	       *(pInc+(k-m_slab_x1.za)*4+2)=ttmp_r*m_sp0[2]-ttmp_i*m_sp0[3];
	       *(pInc+(k-m_slab_x1.za)*4+3)=ttmp_r*m_sp0[3]+ttmp_i*m_sp0[2];
	    }
	 }
      }
   }

   if (m_flgx2Ksi) {
      QPrecision l=n0t1*m_dx;
      QPrecision coef=2.0*m_dtau/l;
      long nz=m_slab_x2.nz;
      long nyz=m_slab_x2.ny*nz;
#pragma omp parallel for num_threads(m_nThread)
      for (long j=m_slab_x2.ya; j<=m_slab_x2.yb; ++j) {
	 QPrecision rj=(j-j0)*dy;
	 QPrecision tmp1=ksi((QPrecision) (j-m_Y1aKsi)/n0t1)
	    *ksi((QPrecision) (m_Y2bKsi-j)/n0t1)*coef;
	 for (long i=m_slab_x2.xa; i<=m_slab_x2.xb; ++i) {
	    QPrecision rij=rj+(i-i0)*dx;
	    QPrecision x=(QPrecision) (m_X2bKsi-i)/n0t1;
	    QPrecision tmp_r=tmp1*d2ksi(x)/l;
	    QPrecision tmp_i=-tmp1*2.0*dksi(x)*sin_theta*cos_phi;
	    QPrecision *pInc=m_Inc_x2+(i-m_slab_x2.xa)*nyz+(j-m_slab_x2.ya)*nz;
#pragma omp simd aligned(pInc:CACHE_LINE)
	    for (long k=m_slab_x2.za; k<=m_slab_x2.zb; ++k) {
	       QPrecision r=rij+(k-k0)*dz;
	       QPrecision cosr=COS(r);
	       QPrecision sinr=SIN(r);
	       QPrecision tmp2=ksi((QPrecision) (k-m_Z1aKsi)/n0t1)
		  *ksi((QPrecision) (m_Z2bKsi-k)/n0t1);
	       QPrecision ttmp_r=tmp2*(tmp_r*cosr-tmp_i*sinr);
	       QPrecision ttmp_i=tmp2*(tmp_r*sinr+tmp_i*cosr);
	       *(pInc+(k-m_slab_x2.za)*4)=ttmp_r*m_sp0[0]-ttmp_i*m_sp0[1];
	       *(pInc+(k-m_slab_x2.za)*4+1)=ttmp_r*m_sp0[1]+ttmp_i*m_sp0[0];
	       *(pInc+(k-m_slab_x2.za)*4+2)=ttmp_r*m_sp0[2]-ttmp_i*m_sp0[3];
	       *(pInc+(k-m_slab_x2.za)*4+3)=ttmp_r*m_sp0[3]+ttmp_i*m_sp0[2];
	    }
	 }
      }
   }

   if (m_flgy1Ksi) {
      QPrecision l=n0t1*m_dy;
      QPrecision coef=2.0*m_dtau/l;
      long nz=m_slab_y1.nz;
      long nyz=m_slab_y1.ny*nz;
#pragma omp parallel for num_threads(m_nThread)
      for (long i=m_slab_y1.xa; i<=m_slab_y1.xb; ++i) {
	 QPrecision ri=(i-i0)*dx;
	 QPrecision tmp1=ksi((QPrecision) (i-m_X1aKsi)/n0t1)
	    *ksi((QPrecision) (m_X2bKsi-i)/n0t1)*coef;
	 for (long j=m_slab_y1.ya; j<=m_slab_y1.yb; ++j) {
	    QPrecision rij=ri+(j-j0)*dy;
	    QPrecision y=(QPrecision) (j-m_Y1aKsi)/n0t1;
	    QPrecision tmp_r=tmp1*d2ksi(y)/l;
	    QPrecision tmp_i=tmp1*2.0*dksi(y)*sin_theta*sin_phi;
	    QPrecision *pInc=m_Inc_y1+(i-m_slab_y1.xa)*nyz+(j-m_slab_y1.ya)*nz;
#pragma omp simd aligned(pInc:CACHE_LINE)
	    for (long k=m_slab_y1.za; k<=m_slab_y1.zb; ++k) {
	       QPrecision r=rij+(k-k0)*dz;
	       QPrecision cosr=COS(r);
	       QPrecision sinr=SIN(r);
	       QPrecision tmp2=ksi((QPrecision) (k-m_Z1aKsi)/n0t1)
		  *ksi((QPrecision) (m_Z2bKsi-k)/n0t1);
	       QPrecision ttmp_r=tmp2*(tmp_r*cosr-tmp_i*sinr);
	       QPrecision ttmp_i=tmp2*(tmp_r*sinr+tmp_i*cosr);
	       *(pInc+(k-m_slab_y1.za)*4)=ttmp_r*m_sp0[0]-ttmp_i*m_sp0[1];
	       *(pInc+(k-m_slab_y1.za)*4+1)=ttmp_r*m_sp0[1]+ttmp_i*m_sp0[0];
	       *(pInc+(k-m_slab_y1.za)*4+2)=ttmp_r*m_sp0[2]-ttmp_i*m_sp0[3];
	       *(pInc+(k-m_slab_y1.za)*4+3)=ttmp_r*m_sp0[3]+ttmp_i*m_sp0[2];
	    }
	 }
      }
   }

   if (m_flgy2Ksi) {
      QPrecision l=n0t1*m_dy;
      QPrecision coef=2.0*m_dtau/l;
      long nz=m_slab_y2.nz;
      long nyz=m_slab_y2.ny*nz;
#pragma omp parallel for num_threads(m_nThread)
      for (long i=m_slab_y2.xa; i<=m_slab_y2.xb; ++i) {
	 QPrecision ri=(i-i0)*dx;
	 QPrecision tmp1=ksi((QPrecision) (i-m_X1aKsi)/n0t1)
	    *ksi((QPrecision) (m_X2bKsi-i)/n0t1)*coef;
	 for (long j=m_slab_y2.ya; j<=m_slab_y2.yb; ++j) {
	    QPrecision rij=ri+(j-j0)*dy;
	    QPrecision y=(QPrecision) (m_Y2bKsi-j)/n0t1;
	    QPrecision tmp_r=tmp1*d2ksi(y)/l;
	    QPrecision tmp_i=-tmp1*2.0*dksi(y)*sin_theta*sin_phi;
	    QPrecision *pInc=m_Inc_y2+(i-m_slab_y2.xa)*nyz+(j-m_slab_y2.ya)*nz;
#pragma omp simd aligned(pInc:CACHE_LINE)
	    for (long k=m_slab_y2.za; k<=m_slab_y2.zb; ++k) {
	       QPrecision r=rij+(k-k0)*dz;
	       QPrecision cosr=COS(r);
	       QPrecision sinr=SIN(r);
	       QPrecision tmp2=ksi((QPrecision) (k-m_Z1aKsi)/n0t1)
		  *ksi((QPrecision) (m_Z2bKsi-k)/n0t1);
	       QPrecision ttmp_r=tmp2*(tmp_r*cosr-tmp_i*sinr);
	       QPrecision ttmp_i=tmp2*(tmp_r*sinr+tmp_i*cosr);
	       *(pInc+(k-m_slab_y2.za)*4)=ttmp_r*m_sp0[0]-ttmp_i*m_sp0[1];
	       *(pInc+(k-m_slab_y2.za)*4+1)=ttmp_r*m_sp0[1]+ttmp_i*m_sp0[0];
	       *(pInc+(k-m_slab_y2.za)*4+2)=ttmp_r*m_sp0[2]-ttmp_i*m_sp0[3];
	       *(pInc+(k-m_slab_y2.za)*4+3)=ttmp_r*m_sp0[3]+ttmp_i*m_sp0[2];
	    }
	 }
      }
   }

   if (m_flgz1Ksi) {
      QPrecision l=n0t1*m_dz;
      QPrecision coef=2.0*m_dtau/l;
      long nz=m_slab_z1.nz;
      long nyz=m_slab_z1.ny*nz;
#pragma omp parallel for num_threads(m_nThread)
      for (long i=m_slab_z1.xa; i<=m_slab_z1.xb; ++i) {
	 QPrecision ri=(i-i0)*dx;
	 QPrecision tmp1=ksi((QPrecision) (i-m_X1aKsi)/n0t1)
	    *ksi((QPrecision) (m_X2bKsi-i)/n0t1)*coef;
	 for (long k=m_slab_z1.za; k<=m_slab_z1.zb; ++k) {
	    QPrecision rik=ri+(k-k0)*dz;
	    QPrecision z=(QPrecision) (k-m_Z1aKsi)/n0t1;
	    QPrecision tmp_r=tmp1*d2ksi(z)/l;
	    QPrecision tmp_i=tmp1*2.0*dksi(z)*cos_theta;
	    QPrecision *pInc=m_Inc_z1+(i-m_slab_z1.xa)*nyz+(k-m_slab_z1.za)*4;
#pragma omp simd
	    for (long j=m_slab_z1.ya; j<=m_slab_z1.yb; ++j) {
	       QPrecision r=rik+(j-j0)*dy;
	       QPrecision cosr=COS(r);
	       QPrecision sinr=SIN(r);
	       QPrecision tmp2=ksi((QPrecision) (j-m_Y1aKsi)/n0t1)
		  *ksi((QPrecision) (m_Y2bKsi-j)/n0t1);
	       QPrecision ttmp_r=tmp2*(tmp_r*cosr-tmp_i*sinr);
	       QPrecision ttmp_i=tmp2*(tmp_r*sinr+tmp_i*cosr);
	       *(pInc+(j-m_slab_z1.ya)*nz)=ttmp_r*m_sp0[0]-ttmp_i*m_sp0[1];
	       *(pInc+(j-m_slab_z1.ya)*nz+1)=ttmp_r*m_sp0[1]+ttmp_i*m_sp0[0];
	       *(pInc+(j-m_slab_z1.ya)*nz+2)=ttmp_r*m_sp0[2]-ttmp_i*m_sp0[3];
	       *(pInc+(j-m_slab_z1.ya)*nz+3)=ttmp_r*m_sp0[3]+ttmp_i*m_sp0[2];
	    }
	 }
      }
   }

   if (m_flgz2Ksi) {
      QPrecision l=n0t1*m_dz;
      QPrecision coef=2.0*m_dtau/l;
      long nz=m_slab_z2.nz;
      long nyz=m_slab_z2.ny*nz;
#pragma omp parallel for num_threads(m_nThread)
      for (long i=m_slab_z2.xa; i<=m_slab_z2.xb; ++i) {
	 QPrecision ri=(i-i0)*dx;
	 QPrecision tmp1=ksi((QPrecision) (i-m_X1aKsi)/n0t1)
	    *ksi((QPrecision) (m_X2bKsi-i)/n0t1)*coef;
	 for (long k=m_slab_z2.za; k<=m_slab_z2.zb; ++k) {
	    QPrecision rik=ri+(k-k0)*dz;
	    QPrecision z=(QPrecision) (m_Z2bKsi-k)/n0t1;
	    QPrecision tmp_r=tmp1*d2ksi(z)/l;
	    QPrecision tmp_i=-tmp1*2.0*dksi(z)*cos_theta;
	    QPrecision *pInc=m_Inc_z2+(i-m_slab_z2.xa)*nyz+(k-m_slab_z2.za)*4;
#pragma omp simd
	    for (long j=m_slab_z2.ya; j<=m_slab_z2.yb; ++j) {
	       QPrecision r=rik+(j-j0)*dy;
	       QPrecision cosr=COS(r);
	       QPrecision sinr=SIN(r);
	       QPrecision tmp2=ksi((QPrecision) (j-m_Y1aKsi)/n0t1)
		  *ksi((QPrecision) (m_Y2bKsi-j)/n0t1);
	       QPrecision ttmp_r=tmp2*(tmp_r*cosr-tmp_i*sinr);
	       QPrecision ttmp_i=tmp2*(tmp_r*sinr+tmp_i*cosr);
	       *(pInc+(j-m_slab_z2.ya)*nz)=ttmp_r*m_sp0[0]-ttmp_i*m_sp0[1];
	       *(pInc+(j-m_slab_z2.ya)*nz+1)=ttmp_r*m_sp0[1]+ttmp_i*m_sp0[0];
	       *(pInc+(j-m_slab_z2.ya)*nz+2)=ttmp_r*m_sp0[2]-ttmp_i*m_sp0[3];
	       *(pInc+(j-m_slab_z2.ya)*nz+3)=ttmp_r*m_sp0[3]+ttmp_i*m_sp0[2];
	    }
	 }
      }
   }
}

// This function only applies to cw wave incidence;
// When under pulsed wave incidence, m_Psi0 and m_Psi are all initialized to 0.
void CPSTD_MagScat::InitPsi()
{
   QPrecision cos_theta=COS(m_theta);
   QPrecision sin_theta=SIN(m_theta);
   QPrecision cos_phi=COS(m_phi);
   QPrecision sin_phi=SIN(m_phi);
   long &i0=m_XInj, &j0=m_YInj, &k0=m_ZInj;
   QPrecision n0t1=(QPrecision) (m_nKsi-1); // m_nKsi grids, (m_nKsi-1) segments
   QPrecision dx=m_dx*sin_theta*cos_phi;
   QPrecision dy=m_dy*sin_theta*sin_phi;
   QPrecision dz=m_dz*cos_theta;
   QPrecision dtc_r=COS(m_dtau);
   QPrecision dtc_i=-SIN(m_dtau);
   // m_Psi0, m_Psi in the total-field and transition layer are initialized
   // to the incident plane wave, 0 in the scattered-field;
   // We will correct their values in the transition zone later.
   QPrecision *psi0=m_Psi0-m_x1FFT*m_nyz-m_y1FFT*m_nz-m_z1FFT*4;
   QPrecision *psi=m_Psi-m_x1FFT*m_nyz-m_y1FFT*m_nz-m_z1FFT*4;
   long ia=(m_x1Core>m_X1aKsi)?m_x1Core:m_X1aKsi;
   long ib=(m_x2Core<m_X2bKsi)?m_x2Core:m_X2bKsi;
   long ja=(m_y1Core>m_Y1aKsi)?m_y1Core:m_Y1aKsi;
   long jb=(m_y2Core<m_Y2bKsi)?m_y2Core:m_Y2bKsi;
   long ka=(m_z1Core>m_Z1aKsi)?m_z1Core:m_Z1aKsi;
   long kb=(m_z2Core<m_Z2bKsi)?m_z2Core:m_Z2bKsi;
#pragma omp parallel for collapse(2) num_threads(m_nThread)
   for (long i=ia; i<=ib; ++i) {
      for (long j=ja; j<=jb; ++j) {
#pragma omp simd
	 for (long k=ka; k<=kb; ++k) {
	    QPrecision r=(QPrecision) (i-i0)*dx + (QPrecision) (j-j0)*dy + (QPrecision) (k-k0)*dz;
	    QPrecision cosr=COS(r);
	    QPrecision sinr=SIN(r);
	    *(psi0+i*m_nyz+j*m_nz+k*4)=cosr*m_sp0[0]-sinr*m_sp0[1];
	    *(psi0+i*m_nyz+j*m_nz+k*4+1)=cosr*m_sp0[1]+sinr*m_sp0[0];
	    *(psi0+i*m_nyz+j*m_nz+k*4+2)=cosr*m_sp0[2]-sinr*m_sp0[3];
	    *(psi0+i*m_nyz+j*m_nz+k*4+3)=cosr*m_sp0[3]+sinr*m_sp0[2];
	    QPrecision ttmp_r=cosr*dtc_r-sinr*dtc_i;
	    QPrecision ttmp_i=cosr*dtc_i+sinr*dtc_r;
	    *(psi+i*m_nyz+j*m_nz+k*4)=ttmp_r*m_sp0[0]-ttmp_i*m_sp0[1];
	    *(psi+i*m_nyz+j*m_nz+k*4+1)=ttmp_r*m_sp0[1]+ttmp_i*m_sp0[0];
	    *(psi+i*m_nyz+j*m_nz+k*4+2)=ttmp_r*m_sp0[2]-ttmp_i*m_sp0[3];
	    *(psi+i*m_nyz+j*m_nz+k*4+3)=ttmp_r*m_sp0[3]+ttmp_i*m_sp0[2];
	 }
      }
   }

   // corret transition zone, back zone
   if (m_flgx1Ksi) {
      // thickness of slab is thin, not enough trip count for m_nThread,
      // placed as outer loop, the i-loop won't be parallelized
      long ibb=(m_x2Core<m_X1bKsi)?m_x2Core:m_X1bKsi;
      for (long i=ia; i<=ibb; ++i) {
	 QPrecision tmp=ksi((QPrecision) (i-m_X1aKsi)/n0t1);
	 QPrecision *p0=psi0+i*m_nyz;
	 QPrecision *p=psi+i*m_nyz;
#pragma omp parallel for num_threads(m_nThread)
	 for (long j=ja; j<=jb; ++j) {
#pragma omp simd
	    for (long k=ka; k<=kb; ++k) {
	       *(p0+j*m_nz+k*4) *= tmp;
	       *(p0+j*m_nz+k*4+1) *= tmp;
	       *(p0+j*m_nz+k*4+2) *= tmp;
	       *(p0+j*m_nz+k*4+3) *= tmp;
	       *(p+j*m_nz+k*4) *= tmp;
	       *(p+j*m_nz+k*4+1) *= tmp;
	       *(p+j*m_nz+k*4+2) *= tmp;
	       *(p+j*m_nz+k*4+3) *= tmp;
	    }
	 }
      }
   }
   // corret transition zone, front zone
   if (m_flgx2Ksi) {
      long iaa=(m_x1Core>m_X2aKsi)?m_x1Core:m_X2aKsi;
      for (long i=iaa; i<=ib; ++i) {
	 QPrecision tmp=ksi((QPrecision) (m_X2bKsi-i)/n0t1);
	 QPrecision *p0=psi0+i*m_nyz;
	 QPrecision *p=psi+i*m_nyz;
#pragma omp parallel for num_threads(m_nThread)
	 for (long j=ja; j<=jb; ++j) {
#pragma omp simd
	    for (long k=ka; k<=kb; ++k) {
	       *(p0+j*m_nz+k*4) *= tmp;
	       *(p0+j*m_nz+k*4+1) *= tmp;
	       *(p0+j*m_nz+k*4+2) *= tmp;
	       *(p0+j*m_nz+k*4+3) *= tmp;
	       *(p+j*m_nz+k*4) *= tmp;
	       *(p+j*m_nz+k*4+1) *= tmp;
	       *(p+j*m_nz+k*4+2) *= tmp;
	       *(p+j*m_nz+k*4+3) *= tmp;
	    }
	 }
      }
   }
   // corret transition zone, left zone
   if (m_flgy1Ksi) {
      long jbb=(m_y2Core<m_Y1bKsi)?m_y2Core:m_Y1bKsi;
      for (long j=ja; j<=jbb; ++j) {
	 QPrecision tmp=ksi((QPrecision) (j-m_Y1aKsi)/n0t1);
	 QPrecision *p0=psi0+j*m_nz;
	 QPrecision *p=psi+j*m_nz;
#pragma omp parallel for num_threads(m_nThread)
	 for (long i=ia; i<=ib; ++i) {
#pragma omp simd
	    for (long k=ka; k<=kb; ++k) {
	       *(p0+i*m_nyz+k*4) *= tmp;
	       *(p0+i*m_nyz+k*4+1) *= tmp;
	       *(p0+i*m_nyz+k*4+2) *= tmp;
	       *(p0+i*m_nyz+k*4+3) *= tmp;
	       *(p+i*m_nyz+k*4) *= tmp;
	       *(p+i*m_nyz+k*4+1) *= tmp;
	       *(p+i*m_nyz+k*4+2) *= tmp;
	       *(p+i*m_nyz+k*4+3) *= tmp;
	    }
	 }
      }
   }
   // corret transition zone, right zone
   if (m_flgy2Ksi) {
      long jaa=(m_y1Core>m_Y2aKsi)?m_y1Core:m_Y2aKsi;
      for (long j=jaa; j<=jb; ++j) {
	 QPrecision tmp=ksi((QPrecision) (m_Y2bKsi-j)/n0t1);
	 QPrecision *p0=psi0+j*m_nz;
	 QPrecision *p=psi+j*m_nz;
#pragma omp parallel for num_threads(m_nThread)
	 for (long i=ia; i<=ib; ++i) {
#pragma omp simd
	    for (long k=ka; k<=kb; ++k) {
	       *(p0+i*m_nyz+k*4) *= tmp;
	       *(p0+i*m_nyz+k*4+1) *= tmp;
	       *(p0+i*m_nyz+k*4+2) *= tmp;
	       *(p0+i*m_nyz+k*4+3) *= tmp;
	       *(p+i*m_nyz+k*4) *= tmp;
	       *(p+i*m_nyz+k*4+1) *= tmp;
	       *(p+i*m_nyz+k*4+2) *= tmp;
	       *(p+i*m_nyz+k*4+3) *= tmp;
	    }
	 }
      }
   }
   // corret transition zone, bottom zone
   if (m_flgz1Ksi) {
      long kbb=(m_z2Core<m_Z1bKsi)?m_z2Core:m_Z1bKsi;
      for (long k=ka; k<=kbb; ++k) {
	 QPrecision tmp=ksi((QPrecision) (k-m_Z1aKsi)/n0t1);
	 QPrecision *p0=psi0+k*4;
	 QPrecision *p=psi+k*4;
#pragma omp parallel for num_threads(m_nThread)
	 for (long i=ia; i<=ib; ++i) {
#pragma omp simd
	    for (long j=ja; j<=jb; ++j) {
	       *(p0+i*m_nyz+j*m_nz) *= tmp;
	       *(p0+i*m_nyz+j*m_nz+1) *= tmp;
	       *(p0+i*m_nyz+j*m_nz+2) *= tmp;
	       *(p0+i*m_nyz+j*m_nz+3) *= tmp;
	       *(p+i*m_nyz+j*m_nz) *= tmp;
	       *(p+i*m_nyz+j*m_nz+1) *= tmp;
	       *(p+i*m_nyz+j*m_nz+2) *= tmp;
	       *(p+i*m_nyz+j*m_nz+3) *= tmp;
	    }
	 }
      }
   }
   // corret transition zone, top zone
   if (m_flgz2Ksi) {
      long kaa=(m_z1Core>m_Z2aKsi)?m_z1Core:m_Z2aKsi;
      for (long k=kaa; k<=kb; ++k) {
	 QPrecision tmp=ksi((QPrecision) (m_Z2bKsi-k)/n0t1);
	 QPrecision *p0=psi0+k*4;
	 QPrecision *p=psi+k*4;
#pragma omp parallel for num_threads(m_nThread)
	 for (long i=ia; i<=ib; ++i) {
#pragma omp simd
	    for (long j=ja; j<=jb; ++j) {
	       *(p0+i*m_nyz+j*m_nz) *= tmp;
	       *(p0+i*m_nyz+j*m_nz+1) *= tmp;
	       *(p0+i*m_nyz+j*m_nz+2) *= tmp;
	       *(p0+i*m_nyz+j*m_nz+3) *= tmp;
	       *(p+i*m_nyz+j*m_nz) *= tmp;
	       *(p+i*m_nyz+j*m_nz+1) *= tmp;
	       *(p+i*m_nyz+j*m_nz+2) *= tmp;
	       *(p+i*m_nyz+j*m_nz+3) *= tmp;
	    }
	 }
      }
   }

   ExchangeData();
   m_nTau=1; // We have initialized Psi0 to time step 0, Psi to time step 1
}

// initialize the potential grids
void CPSTD_MagScat::InitV(void (*func)(QPrecision *, QPrecision, QPrecision, QPrecision, QPrecision))
{
   if (!m_flgV) return;
   QPrecision *pBase=m_VB_i-(m_x1V*m_nyzV+m_y1V*m_nzV+m_z1V*4);
   QPrecision v0=2.0*m_dtau/m_E0;
   QPrecision VB[4];

   QPrecision r_cutoff=FABS(m_OrigX-(m_X1bKsi+1))*m_dx;
   QPrecision tmp=FABS(m_OrigX-(m_X2aKsi-1))*m_dx;
   if (r_cutoff>tmp) r_cutoff=tmp;
   tmp=FABS(m_OrigY-(m_Y1bKsi+1))*m_dy;
   if (r_cutoff>tmp) r_cutoff=tmp;
   tmp=FABS(m_OrigY-(m_Y2aKsi-1))*m_dy;
   if (r_cutoff>tmp) r_cutoff=tmp;
   tmp=FABS(m_OrigZ-(m_Z1bKsi+1))*m_dz;
   if (r_cutoff>tmp) r_cutoff=tmp;
   tmp=FABS(m_OrigZ-(m_Z2aKsi-1))*m_dz;
   if (r_cutoff>tmp) r_cutoff=tmp;

#pragma omp parallel for num_threads(m_nThread)
   for (long i=m_x1V; i<=m_x2V; ++i) {
      // QPrecision x=(i-m_OrigX)*m_dx*m_lambdabar0;
      QPrecision x=(i-m_OrigX)*m_dx;	// we already set m_lambdabar0 as the length unit in the potential function
      for (long j=m_y1V; j<=m_y2V; ++j) {
	 // QPrecision y=(j-m_OrigY)*m_dy*m_lambdabar0;
	 QPrecision y=(j-m_OrigY)*m_dy;
	 // no simd here, because the potential function is a branch
	 for (long k=m_z1V; k<=m_z2V; ++k) {
	    // QPrecision z=(k-m_OrigZ)*m_dz*m_lambdabar0;
	    QPrecision z=(k-m_OrigZ)*m_dz;
	    (*func)(VB,x,y,z,r_cutoff);
	    *(pBase+i*m_nyzV+j*m_nzV+k*4)=VB[0]*v0;	// reserved for scalar potential; not used in this code
	    *(pBase+i*m_nyzV+j*m_nzV+k*4+1)=VB[1]*v0;	// potential by Bx
	    *(pBase+i*m_nyzV+j*m_nzV+k*4+2)=VB[2]*v0;	// potential by By
	    *(pBase+i*m_nyzV+j*m_nzV+k*4+3)=VB[3]*v0;	// potential by Bz
	 }
      }
   }
}

void CPSTD_MagScat::InitVirtualSurfaces()
{
   m_nSur=0;
   m_flgx1Vrtl=(m_x1Core<=m_X1Vrtl && m_X1Vrtl<=m_x2Core)?true:false;
   m_flgx2Vrtl=(m_x1Core<=m_X2Vrtl && m_X2Vrtl<=m_x2Core)?true:false;
   m_flgy1Vrtl=(m_y1Core<=m_Y1Vrtl && m_Y1Vrtl<=m_y2Core)?true:false;
   m_flgy2Vrtl=(m_y1Core<=m_Y2Vrtl && m_Y2Vrtl<=m_y2Core)?true:false;
   m_flgz1Vrtl=(m_z1Core<=m_Z1Vrtl && m_Z1Vrtl<=m_z2Core)?true:false;
   m_flgz2Vrtl=(m_z1Core<=m_Z2Vrtl && m_Z2Vrtl<=m_z2Core)?true:false;

   if (m_flgx1Vrtl) {
      m_Surf_x_ny=m_nyCore;
      m_Surf_x_nz=m_nzCore*4;

      int val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfPsi_x1, m_Surf_x_ny, m_Surf_x_nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfPsi_x1");

      val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_PS_x1, m_Surf_x_ny, m_Surf_x_nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfDPsi_PS_x1");

      val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_FD_x1, m_Surf_x_ny, m_Surf_x_nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfDPsi_FD_x1");
   }
   if (m_flgx2Vrtl) {
      m_Surf_x_ny=m_nyCore;
      m_Surf_x_nz=m_nzCore*4;

      int val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfPsi_x2, m_Surf_x_ny, m_Surf_x_nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfPsi_x2");

      val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_PS_x2, m_Surf_x_ny, m_Surf_x_nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfDPsi_PS_x2");

      val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_FD_x2, m_Surf_x_ny, m_Surf_x_nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfDPsi_FD_x2");
   }
   if (m_flgy1Vrtl) {
      m_Surf_y_nx=m_nxCore;
      m_Surf_y_nz=m_nzCore*4;

      int val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfPsi_y1, m_Surf_y_nx, m_Surf_y_nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfPsi_y1");

      val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_PS_y1, m_Surf_y_nx, m_Surf_y_nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfDPsi_PS_y1");

      val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_FD_y1, m_Surf_y_nx, m_Surf_y_nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfDPsi_FD_y1");
   }
   if (m_flgy2Vrtl) {
      m_Surf_y_nx=m_nxCore;
      m_Surf_y_nz=m_nzCore*4;

      int val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfPsi_y2, m_Surf_y_nx, m_Surf_y_nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfPsi_y2");

      val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_PS_y2, m_Surf_y_nx, m_Surf_y_nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfDPsi_PS_y2");

      val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_FD_y2, m_Surf_y_nx, m_Surf_y_nz, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfDPsi_FD_y2");
   }
   if (m_flgz1Vrtl) {
      m_Surf_z_nx=m_nxCore;
      m_Surf_z_ny=m_nyCore*4;

      int val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfPsi_z1, m_Surf_z_nx, m_Surf_z_ny, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfPsi_z1");

      val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_PS_z1, m_Surf_z_nx, m_Surf_z_ny, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfDPsi_PS_z1");

      val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_FD_z1, m_Surf_z_nx, m_Surf_z_ny, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfDPsi_FD_z1");
   }
   if (m_flgz2Vrtl) {
      m_Surf_z_nx=m_nxCore;
      m_Surf_z_ny=m_nyCore*4;

      int val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfPsi_z2, m_Surf_z_nx, m_Surf_z_ny, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfPsi_z2");

      val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_PS_z2, m_Surf_z_nx, m_Surf_z_ny, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfDPsi_PS_z2");

      val=Init_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_FD_z2, m_Surf_z_nx, m_Surf_z_ny, sizeof(QPrecision)*4);
      if (val) HandleError(val, "cannot allocate m_SurfDPsi_FD_z2");
   }
}

// This will continue the computation from a previous run, by loading the saved 
// snapshot of the memory when the run was about to end.  This faciliates the 
// calculation when one finds the time iterations of the previous run are insufficient.
// Also when working on a supercomputer, there is a rate the machine can go down.  Our
// account manager at the supercomputing center has asked us to save the middle results 
// periodically.
// Note: MPILoadSnapshot_all only loads the data saved with MPISaveSnapshot_all;
// However, it turns out that MPILoadSnapshot_all loads data very slowly.  That
// is why we make another pair of member functions, MPILoadSnapshot and MPISaveSnapshot.
void CPSTD_MagScat::MPILoadSnapshot_all(string snapshot,
      void (*func) (QPrecision *, QPrecision, QPrecision, QPrecision, QPrecision),
      MPI_Comm comm)
{
   MPI_File fh;
   if (MPI_File_open(comm, snapshot.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)!=MPI_SUCCESS)
      throw runtime_error("Cannot open file " + snapshot);

   int rank, nx_proc, ny_proc, nz_proc;
   MPI_Comm_rank(comm, &rank);
   if (!rank) {
      stringstream sstr;
      int precision;
      MPI_File_read(fh, &precision, 1, MPIInt, MPI_STATUS_IGNORE);
      if (precision!=sizeof(QPrecision)) {
	 sstr << "Error: Data precision and App precision mismatch -- "
	    << precision << "vs. " << sizeof(QPrecision);
	 throw runtime_error(sstr.str());
      }

      MPI_File_read(fh, &nx_proc, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &ny_proc, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &nz_proc, 1, MPIInt, MPI_STATUS_IGNORE);
   }

   MPI_Bcast(&nx_proc, 1, MPIInt, 0, comm);
   MPI_Bcast(&ny_proc, 1, MPIInt, 0, comm);
   MPI_Bcast(&nz_proc, 1, MPIInt, 0, comm);
   SetupCartComm(nx_proc, ny_proc, nz_proc, comm);

   if (!rank) {
      MPI_File_read(fh, &m_nXFFT, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nYFFT, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nZFFT, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nABC, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nSca, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nKsi, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nHalo, 1, MPILong, MPI_STATUS_IGNORE);
   }

   MPI_Bcast(&m_nXFFT, 1, MPILong, 0, comm);
   MPI_Bcast(&m_nYFFT, 1, MPILong, 0, comm);
   MPI_Bcast(&m_nZFFT, 1, MPILong, 0, comm);
   MPI_Bcast(&m_nABC, 1, MPILong, 0, comm);
   MPI_Bcast(&m_nSca, 1, MPILong, 0, comm);
   MPI_Bcast(&m_nKsi, 1, MPILong, 0, comm);
   MPI_Bcast(&m_nHalo, 1, MPILong, 0, comm);
   DomainDecomposition();
   InitMPIExchange();

   if (!rank) {
      MPI_File_read(fh, &m_dtau, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_dx, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_dy, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_dz, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_theta, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_phi, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_E0, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, m_sp0, 4, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_CAP_U0, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_CAP_alpha, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_OrigX, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_OrigY, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_OrigZ, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, m_weight, NWEIGHT, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nTau, 1, MPILong, MPI_STATUS_IGNORE);
   }

   MPI_Bcast(&m_dtau, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_dx, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_dy, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_dz, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_theta, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_phi, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_E0, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(m_sp0, 4, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_CAP_U0, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_CAP_alpha, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_OrigX, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_OrigY, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_OrigZ, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(m_weight, NWEIGHT, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_nTau, 1, MPILong, 0, comm);

   m_lambdabar0=MICROEV2NMBAR/SQRT(m_E0);
   m_alpha_2dtau_i=2.0*SIN(m_dtau);

   MPI_Offset disp=sizeof(int)+sizeof(m_nXProcs)+sizeof(m_nYProcs)+sizeof(m_nZProcs)
      +sizeof(m_nXFFT)+sizeof(m_nYFFT)+sizeof(m_nZFFT)+sizeof(m_nABC)+sizeof(m_nSca)
      +sizeof(m_nKsi)+sizeof(m_nHalo)+sizeof(m_dtau)+sizeof(m_dx)+sizeof(m_dy)+sizeof(m_dz)
      +sizeof(m_theta)+sizeof(m_phi)+sizeof(m_E0)+sizeof(m_sp0)
      +sizeof(m_CAP_U0)+sizeof(m_CAP_alpha)
      +sizeof(m_OrigX)+sizeof(m_OrigY)+sizeof(m_OrigZ)+NWEIGHT*sizeof(QPrecision)
      +sizeof(m_nTau);

   MPI_Datatype global_array, local_array;
   int gsizes[3], lsizes[3], subsizes[3], starts[3];
   lsizes[0]=m_nx; lsizes[1]=m_ny; lsizes[2]=m_nz;
   subsizes[0]=m_nxCore; subsizes[1]=m_nyCore; subsizes[2]=m_nzCore*4;
   starts[0]=m_x1Core-m_x1FFT; starts[1]=m_y1Core-m_y1FFT; starts[2]=(m_z1Core-m_z1FFT)*4;
   MPI_Type_create_subarray(3, lsizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &local_array);
   MPI_Type_commit(&local_array);
   gsizes[0]=m_nXGlb; gsizes[1]=m_nYGlb; gsizes[2]=m_nZGlb*4;
   starts[0]=m_x1Core; starts[1]=m_y1Core; starts[2]=m_z1Core*4;
   MPI_Type_create_subarray(3, gsizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &global_array);
   MPI_Type_commit(&global_array);
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_read_all(fh, m_Psi0, 1, local_array, MPI_STATUS_IGNORE);
   disp += gsizes[0]*gsizes[1]*gsizes[2]*sizeof(QPrecision);
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_read_all(fh, m_Psi, 1, local_array, MPI_STATUS_IGNORE);
   MPI_Type_free(&local_array);
   MPI_Type_free(&global_array);
   MPI_File_close(&fh);

   InitFFT();
   InitABC();
   InitTS();
   ExchangeData();
   InitV(func);
   InitVirtualSurfaces();
   MPI_Barrier(m_cartcomm);

#ifdef __INIT_STATUS__
   WriteInitStatusToSTDOUT();
#endif
}

void CPSTD_MagScat::MPISaveSnapshot_all(string snapshot_prefix)
{
   MPI_File fh;
   stringstream sstr;
   sstr << snapshot_prefix << m_nTau << "_all.dat";
   if (MPI_File_open(m_cartcomm, sstr.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY,
	    MPI_INFO_NULL, &fh) != MPI_SUCCESS)
      throw runtime_error("Cannot open output file " + sstr.str());

   if (!m_Rank) {
      int precision=sizeof(QPrecision);
      MPI_File_write(fh, &precision, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nXProcs, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nYProcs, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nZProcs, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nXFFT, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nYFFT, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nZFFT, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nABC, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nSca, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nKsi, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nHalo, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dtau, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dx, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dy, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dz, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_theta, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_phi, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_E0, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, m_sp0, 4, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_CAP_U0, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_CAP_alpha, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_OrigX, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_OrigY, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_OrigZ, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, m_weight, NWEIGHT, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nTau, 1, MPILong, MPI_STATUS_IGNORE);
   }

   MPI_Offset disp=sizeof(int)+sizeof(m_nXProcs)+sizeof(m_nYProcs)+sizeof(m_nZProcs)
      +sizeof(m_nXFFT)+sizeof(m_nYFFT)+sizeof(m_nZFFT)+sizeof(m_nABC)+sizeof(m_nSca)
      +sizeof(m_nKsi)+sizeof(m_nHalo)+sizeof(m_dtau)+sizeof(m_dx)+sizeof(m_dy)+sizeof(m_dz)
      +sizeof(m_theta)+sizeof(m_phi)+sizeof(m_E0)+sizeof(m_sp0)
      +sizeof(m_CAP_U0)+sizeof(m_CAP_alpha)
      +sizeof(m_OrigX)+sizeof(m_OrigY)+sizeof(m_OrigZ)
      +NWEIGHT*sizeof(QPrecision)+sizeof(m_nTau);

   MPI_Datatype global_array, local_array;
   int gsizes[3], lsizes[3], subsizes[3], starts[3];
   lsizes[0]=m_nx; lsizes[1]=m_ny; lsizes[2]=m_nz;
   subsizes[0]=m_nxCore; subsizes[1]=m_nyCore; subsizes[2]=m_nzCore*4;
   starts[0]=m_x1Core-m_x1FFT; starts[1]=m_y1Core-m_y1FFT; starts[2]=(m_z1Core-m_z1FFT)*4;
   MPI_Type_create_subarray(3, lsizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &local_array);
   MPI_Type_commit(&local_array);
   gsizes[0]=m_nXGlb; gsizes[1]=m_nYGlb; gsizes[2]=m_nZGlb*4;
   starts[0]=m_x1Core; starts[1]=m_y1Core; starts[2]=m_z1Core*4;
   MPI_Type_create_subarray(3, gsizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &global_array);
   MPI_Type_commit(&global_array);
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, m_Psi0, 1, local_array, MPI_STATUS_IGNORE);
   disp += gsizes[0]*gsizes[1]*gsizes[2]*sizeof(QPrecision); 
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, m_Psi, 1, local_array, MPI_STATUS_IGNORE);
   MPI_Type_free(&local_array);
   MPI_Type_free(&global_array);
   MPI_File_close(&fh);
}

// This also facilitates the contious executation of a previous run, except this function loads
// data much faster.  The data saved by MPISaveSnapshot contains redundant ones from the halo
// region.  (wasted CPU time vs wasted storage space)
void CPSTD_MagScat::MPILoadSnapshot(string snapshot,
      void (*func) (QPrecision *, QPrecision, QPrecision, QPrecision, QPrecision),
      MPI_Comm comm)
{
   MPI_File fh;
   if (MPI_File_open(comm, snapshot.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)!=MPI_SUCCESS)
      throw runtime_error("Cannot open file " + snapshot);

   int rank, nx_proc, ny_proc, nz_proc;
   MPI_Comm_rank(comm, &rank);
   if (!rank) {
      stringstream sstr;
      int precision;
      MPI_File_read(fh, &precision, 1, MPIInt, MPI_STATUS_IGNORE);
      if (precision!=sizeof(QPrecision)) {
	 sstr << "Error: Data precision and App precision mismatch -- "
	    << precision << "vs. " << sizeof(QPrecision);
	 throw runtime_error(sstr.str());
      }

      MPI_File_read(fh, &nx_proc, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &ny_proc, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &nz_proc, 1, MPIInt, MPI_STATUS_IGNORE);
   }

   MPI_Bcast(&nx_proc, 1, MPIInt, 0, comm);
   MPI_Bcast(&ny_proc, 1, MPIInt, 0, comm);
   MPI_Bcast(&nz_proc, 1, MPIInt, 0, comm);
   SetupCartComm(nx_proc, ny_proc, nz_proc, comm);

   if (!rank) {
      MPI_File_read(fh, &m_nXFFT, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nYFFT, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nZFFT, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nABC, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nSca, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nKsi, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nHalo, 1, MPILong, MPI_STATUS_IGNORE);
   }

   MPI_Bcast(&m_nXFFT, 1, MPILong, 0, comm);
   MPI_Bcast(&m_nYFFT, 1, MPILong, 0, comm);
   MPI_Bcast(&m_nZFFT, 1, MPILong, 0, comm);
   MPI_Bcast(&m_nABC, 1, MPILong, 0, comm);
   MPI_Bcast(&m_nSca, 1, MPILong, 0, comm);
   MPI_Bcast(&m_nKsi, 1, MPILong, 0, comm);
   MPI_Bcast(&m_nHalo, 1, MPILong, 0, comm);
   DomainDecomposition();
   InitMPIExchange();

   if (!rank) {
      MPI_File_read(fh, &m_dtau, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_dx, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_dy, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_dz, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_theta, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_phi, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_E0, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, m_sp0, 4, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_CAP_U0, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_CAP_alpha, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_OrigX, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_OrigY, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_OrigZ, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, m_weight, NWEIGHT, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_read(fh, &m_nTau, 1, MPILong, MPI_STATUS_IGNORE);
   }

   MPI_Bcast(&m_dtau, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_dx, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_dy, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_dz, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_theta, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_phi, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_E0, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(m_sp0, 4, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_CAP_U0, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_CAP_alpha, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_OrigX, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_OrigY, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_OrigZ, 1, MPIQPrecision, 0, comm);
   MPI_Bcast(m_weight, NWEIGHT, MPIQPrecision, 0, comm);
   MPI_Bcast(&m_nTau, 1, MPILong, 0, comm);

   m_lambdabar0=MICROEV2NMBAR/SQRT(m_E0);
   m_alpha_2dtau_i=2.0*SIN(m_dtau);

   MPI_Offset disp=sizeof(int)+sizeof(m_nXProcs)+sizeof(m_nYProcs)+sizeof(m_nZProcs)
      +sizeof(m_nXFFT)+sizeof(m_nYFFT)+sizeof(m_nZFFT)+sizeof(m_nABC)+sizeof(m_nSca)
      +sizeof(m_nKsi)+sizeof(m_nHalo)+sizeof(m_dtau)+sizeof(m_dx)+sizeof(m_dy)+sizeof(m_dz)
      +sizeof(m_theta)+sizeof(m_phi)+sizeof(m_E0)+sizeof(m_sp0)
      +sizeof(m_CAP_U0)+sizeof(m_CAP_alpha)
      +sizeof(m_OrigX)+sizeof(m_OrigY)+sizeof(m_OrigZ)+m_nHalo*sizeof(QPrecision)
      +sizeof(m_nTau);

   disp+=(long) ((m_xRank*m_nYProcs+m_yRank)*m_nZProcs+m_zRank) * m_nx*m_nyz*sizeof(QPrecision);
   MPI_File_read_at(fh, disp, m_Psi0, m_nx*m_nyz, MPIQPrecision, MPI_STATUS_IGNORE);
   disp+=(long) (m_nXProcs*m_nYProcs*m_nZProcs) * m_nx*m_nyz*sizeof(QPrecision);
   MPI_File_read_at(fh, disp, m_Psi, m_nx*m_nyz, MPIQPrecision, MPI_STATUS_IGNORE);
   MPI_File_close(&fh);

   InitFFT();
   InitABC();
   InitTS();
   ExchangeData();
   InitV(func);
   InitVirtualSurfaces();
   MPI_Barrier(m_cartcomm);

#ifdef __INIT_STATUS__
   WriteInitStatusToSTDOUT();
#endif
}

void CPSTD_MagScat::MPISaveSnapshot(string snapshot_prefix)
{
   MPI_File fh;
   stringstream sstr;
   sstr << snapshot_prefix << m_nTau << ".dat";
   if (MPI_File_open(m_cartcomm, sstr.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY,
	    MPI_INFO_NULL, &fh) != MPI_SUCCESS)
      throw runtime_error("Cannot open output file " + sstr.str());

   if (!m_Rank) {
      int precision=sizeof(QPrecision);
      MPI_File_write(fh, &precision, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nXProcs, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nYProcs, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nZProcs, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nXFFT, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nYFFT, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nZFFT, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nABC, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nSca, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nKsi, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nHalo, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dtau, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dx, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dy, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dz, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_theta, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_phi, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_E0, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, m_sp0, 4, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_CAP_U0, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_CAP_alpha, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_OrigX, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_OrigY, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_OrigZ, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, m_weight, NWEIGHT, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nTau, 1, MPILong, MPI_STATUS_IGNORE);
   }

   MPI_Offset disp=sizeof(int)+sizeof(m_nXProcs)+sizeof(m_nYProcs)+sizeof(m_nZProcs)
      +sizeof(m_nXFFT)+sizeof(m_nYFFT)+sizeof(m_nZFFT)+sizeof(m_nABC)+sizeof(m_nSca)
      +sizeof(m_nKsi)+sizeof(m_nHalo)+sizeof(m_dtau)+sizeof(m_dx)+sizeof(m_dy)+sizeof(m_dz)
      +sizeof(m_theta)+sizeof(m_phi)+sizeof(m_E0)+sizeof(m_sp0)
      +sizeof(m_CAP_U0)+sizeof(m_CAP_alpha)
      +sizeof(m_OrigX)+sizeof(m_OrigY)+sizeof(m_OrigZ)
      +NWEIGHT*sizeof(QPrecision)+sizeof(m_nTau);

   disp+=(long) ((m_xRank*m_nYProcs+m_yRank)*m_nZProcs+m_zRank) * m_nx*m_nyz*sizeof(QPrecision);
   MPI_File_write_at(fh, disp, m_Psi0, m_nx*m_nyz, MPIQPrecision, MPI_STATUS_IGNORE);
   disp+=(long) (m_nXProcs*m_nYProcs*m_nZProcs) * m_nx*m_nyz*sizeof(QPrecision);
   MPI_File_write_at(fh, disp, m_Psi, m_nx*m_nyz, MPIQPrecision, MPI_STATUS_IGNORE);
   MPI_File_close(&fh);
}

void CPSTD_MagScat::Update()
{
   QPrecision t=m_nTau*m_dtau;
   QPrecision factor_introduce_V=t/TWOPI;	// using one cycle to introduce V
   // plane wave time factor exp(-iEt/hbar), multiplied by I
   QPrecision Iinc_r=SIN((QPrecision) t);
   QPrecision Iinc_i=COS((QPrecision) t);

#pragma omp parallel default(shared) num_threads(m_nThread)
   {
   // the $\frac{\partial^2}{\partial x^2}$ term
#pragma omp for collapse(2)
      for (long j=m_j1Core; j<=m_j2Core; ++j) {
	 for (long k=m_k1Core; k<=m_k2Core; ++k) {
	    UpdateX(omp_get_thread_num(), j, k);
	 }
      }

   // the $\frac{\partial^2}{\partial y^2}$ term
#pragma omp for collapse(2)
      for (long i=m_i1Core; i<=m_i2Core; ++i) {
	 for (long k=m_k1Core; k<=m_k2Core; ++k) {
	    UpdateY(omp_get_thread_num(), i, k);
	 }
      }

   // the $\frac{\partial^2}{\partial y^2}$ term
#pragma omp for collapse(2)
      for (long i=m_i1Core; i<=m_i2Core; ++i) {
	 for (long j=m_j1Core; j<=m_j2Core; ++j) {
	    UpdateZ(omp_get_thread_num(), i, j);
	 }
      }

      if (m_flgV) {
	 // During the first time cycle, we gradually increase the potential to the model.
         // This is only applicable to the case of cw wave incidence.
         // Remove the following if-branch when pulsed incidence is used.
	 if (factor_introduce_V<0.997) {
#pragma omp for collapse(2)
	    for (long i=m_x1V; i<=m_x2V; ++i) {
	       for (long j=m_y1V; j<=m_y2V; ++j) {
		  QPrecision *pV=m_VB_i+(i-m_x1V)*m_nyzV+(j-m_y1V)*m_nzV;
		  QPrecision *psi=m_Psi+(i-m_x1FFT)*m_nyz+(j-m_y1FFT)*m_nz;
		  QPrecision *psi0=m_Psi0+(i-m_x1FFT)*m_nyz+(j-m_y1FFT)*m_nz;
#pragma omp simd aligned(psi0:CACHE_LINE) aligned(psi:CACHE_LINE) aligned(pV:CACHE_LINE)
		  for (long k=m_z1V; k<=m_z2V; ++k) {
		     long ktmp=(k-m_z1V)*4;
		     QPrecision v1=*(pV+ktmp+1);
		     QPrecision v2=*(pV+ktmp+2);
		     QPrecision v3=*(pV+ktmp+3);
		     ktmp=(k-m_z1FFT)*4;
		     QPrecision p0=*(psi+ktmp);
		     QPrecision p1=*(psi+ktmp+1);
		     QPrecision p2=*(psi+ktmp+2);
		     QPrecision p3=*(psi+ktmp+3);
		     *(psi0+ktmp)   += factor_introduce_V*(v3*p1-v2*p2+v1*p3);
		     *(psi0+ktmp+1) += factor_introduce_V*(-v3*p0-v1*p2-v2*p3);
		     *(psi0+ktmp+2) += factor_introduce_V*(v2*p0+v1*p1-v3*p3);
		     *(psi0+ktmp+3) += factor_introduce_V*(-v1*p0+v2*p1+v3*p2);
		  }
	       }
	    }
	 } else {
#pragma omp for collapse(2)
	    for (long i=m_x1V; i<=m_x2V; ++i) {
	       for (long j=m_y1V; j<=m_y2V; ++j) {
		  QPrecision *pV=m_VB_i+(i-m_x1V)*m_nyzV+(j-m_y1V)*m_nzV;
		  QPrecision *psi=m_Psi+(i-m_x1FFT)*m_nyz+(j-m_y1FFT)*m_nz;
		  QPrecision *psi0=m_Psi0+(i-m_x1FFT)*m_nyz+(j-m_y1FFT)*m_nz;
#pragma omp simd aligned(psi0:CACHE_LINE) aligned(psi:CACHE_LINE) aligned(pV:CACHE_LINE)
		  for (long k=m_z1V; k<=m_z2V; ++k) {
		     long ktmp=(k-m_z1V)*4;
		     QPrecision v1=*(pV+ktmp+1);
		     QPrecision v2=*(pV+ktmp+2);
		     QPrecision v3=*(pV+ktmp+3);
		     ktmp=(k-m_z1FFT)*4;
		     QPrecision p0=*(psi+ktmp);
		     QPrecision p1=*(psi+ktmp+1);
		     QPrecision p2=*(psi+ktmp+2);
		     QPrecision p3=*(psi+ktmp+3);
		     *(psi0+ktmp)   += v3*p1-v2*p2+v1*p3;
		     *(psi0+ktmp+1) += -v3*p0-v1*p2-v2*p3;
		     *(psi0+ktmp+2) += v2*p0+v1*p1-v3*p3;
		     *(psi0+ktmp+3) += -v1*p0+v2*p1+v3*p2;
		  }
	       }
	    }
	 }
      }

      // incidence term: total-field/scattered-field at the back transition-layer
      if (m_flgx1Ksi) {
	 long nz=m_slab_x1.nz;
	 long nyz=m_slab_x1.ny*nz;
#pragma omp for collapse(2)
	 for (long i=m_slab_x1.xa; i<=m_slab_x1.xb; ++i) {
	    for (long j=m_slab_x1.ya; j<=m_slab_x1.yb; ++j) {
	       QPrecision *pInc=m_Inc_x1+(i-m_slab_x1.xa)*nyz+(j-m_slab_x1.ya)*nz;
	       QPrecision *psi0=m_Psi0+(i-m_x1FFT)*m_nyz+(j-m_y1FFT)*m_nz;
#pragma omp simd aligned(psi0:CACHE_LINE) aligned(pInc:CACHE_LINE)
	       for (long k=m_slab_x1.za; k<=m_slab_x1.zb; ++k) {
		  long ktmp1=(k-m_slab_x1.za)*4;
		  long ktmp2=(k-m_z1FFT)*4;
		  *(psi0+ktmp2) -= (*(pInc+ktmp1))*Iinc_r-(*(pInc+ktmp1+1))*Iinc_i;
		  *(psi0+ktmp2+1) -= (*(pInc+ktmp1))*Iinc_i+(*(pInc+ktmp1+1))*Iinc_r;
		  *(psi0+ktmp2+2) -= (*(pInc+ktmp1+2))*Iinc_r-(*(pInc+ktmp1+3))*Iinc_i;
		  *(psi0+ktmp2+3) -= (*(pInc+ktmp1+2))*Iinc_i+(*(pInc+ktmp1+3))*Iinc_r;
	       }
	    }
	 }
      }

      // incidence term: total-field/scattered-field at the front transition-layer
      if (m_flgx2Ksi) {
	 long nz=m_slab_x2.nz;
	 long nyz=m_slab_x2.ny*nz;
#pragma omp for collapse(2)
	 for (long i=m_slab_x2.xa; i<=m_slab_x2.xb; ++i) {
	    for (long j=m_slab_x2.ya; j<=m_slab_x2.yb; ++j) {
	       QPrecision *pInc=m_Inc_x2+(i-m_slab_x2.xa)*nyz+(j-m_slab_x2.ya)*nz;
	       QPrecision *psi0=m_Psi0+(i-m_x1FFT)*m_nyz+(j-m_y1FFT)*m_nz;
#pragma omp simd aligned(psi0:CACHE_LINE) aligned(pInc:CACHE_LINE)
	       for (long k=m_slab_x2.za; k<=m_slab_x2.zb; ++k) {
		  long ktmp1=(k-m_slab_x2.za)*4;
		  long ktmp2=(k-m_z1FFT)*4;
		  *(psi0+ktmp2) -= (*(pInc+ktmp1))*Iinc_r-(*(pInc+ktmp1+1))*Iinc_i;
		  *(psi0+ktmp2+1) -= (*(pInc+ktmp1))*Iinc_i+(*(pInc+ktmp1+1))*Iinc_r;
		  *(psi0+ktmp2+2) -= (*(pInc+ktmp1+2))*Iinc_r-(*(pInc+ktmp1+3))*Iinc_i;
		  *(psi0+ktmp2+3) -= (*(pInc+ktmp1+2))*Iinc_i+(*(pInc+ktmp1+3))*Iinc_r;
	       }
	    }
	 }
      }

      // incidence term: total-field/scattered-field at the left transition-layer
      if (m_flgy1Ksi) {
	 long nz=m_slab_y1.nz;
	 long nyz=m_slab_y1.ny*nz;
#pragma omp for collapse(2)
	 for (long i=m_slab_y1.xa; i<=m_slab_y1.xb; ++i) {
	    for (long j=m_slab_y1.ya; j<=m_slab_y1.yb; ++j) {
	       QPrecision *pInc=m_Inc_y1+(i-m_slab_y1.xa)*nyz+(j-m_slab_y1.ya)*nz;
	       QPrecision *psi0=m_Psi0+(i-m_x1FFT)*m_nyz+(j-m_y1FFT)*m_nz;
#pragma omp simd aligned(psi0:CACHE_LINE) aligned(pInc:CACHE_LINE)
	       for (long k=m_slab_y1.za; k<=m_slab_y1.zb; ++k) {
		  long ktmp1=(k-m_slab_y1.za)*4;
		  long ktmp2=(k-m_z1FFT)*4;
		  *(psi0+ktmp2) -= (*(pInc+ktmp1))*Iinc_r-(*(pInc+ktmp1+1))*Iinc_i;
		  *(psi0+ktmp2+1) -= (*(pInc+ktmp1))*Iinc_i+(*(pInc+ktmp1+1))*Iinc_r;
		  *(psi0+ktmp2+2) -= (*(pInc+ktmp1+2))*Iinc_r-(*(pInc+ktmp1+3))*Iinc_i;
		  *(psi0+ktmp2+3) -= (*(pInc+ktmp1+2))*Iinc_i+(*(pInc+ktmp1+3))*Iinc_r;
	       }
	    }
	 }
      }

      // incidence term: total-field/scattered-field at the right transition-layer
      if (m_flgy2Ksi) {
	 long nz=m_slab_y2.nz;
	 long nyz=m_slab_y2.ny*nz;
#pragma omp for collapse(2)
	 for (long i=m_slab_y2.xa; i<=m_slab_y2.xb; ++i) {
	    for (long j=m_slab_y2.ya; j<=m_slab_y2.yb; ++j) {
	       QPrecision *pInc=m_Inc_y2+(i-m_slab_y2.xa)*nyz+(j-m_slab_y2.ya)*nz;
	       QPrecision *psi0=m_Psi0+(i-m_x1FFT)*m_nyz+(j-m_y1FFT)*m_nz;
#pragma omp simd aligned(psi0:CACHE_LINE) aligned(pInc:CACHE_LINE)
	       for (long k=m_slab_y2.za; k<=m_slab_y2.zb; ++k) {
		  long ktmp1=(k-m_slab_y2.za)*4;
		  long ktmp2=(k-m_z1FFT)*4;
		  *(psi0+ktmp2) -= (*(pInc+ktmp1))*Iinc_r-(*(pInc+ktmp1+1))*Iinc_i;
		  *(psi0+ktmp2+1) -= (*(pInc+ktmp1))*Iinc_i+(*(pInc+ktmp1+1))*Iinc_r;
		  *(psi0+ktmp2+2) -= (*(pInc+ktmp1+2))*Iinc_r-(*(pInc+ktmp1+3))*Iinc_i;
		  *(psi0+ktmp2+3) -= (*(pInc+ktmp1+2))*Iinc_i+(*(pInc+ktmp1+3))*Iinc_r;
	       }
	    }
	 }
      }

      // incidence term: total-field/scattered-field at the bottom transition-layer
      if (m_flgz1Ksi) {
	 long nz=m_slab_z1.nz;
	 long nyz=m_slab_z1.ny*nz;
#pragma omp for collapse(2)
	 for (long i=m_slab_z1.xa; i<=m_slab_z1.xb; ++i) {
	    for (long k=m_slab_z1.za; k<=m_slab_z1.zb; ++k) {
	       QPrecision *pInc=m_Inc_z1+(i-m_slab_z1.xa)*nyz+(k-m_slab_z1.za)*4;
	       QPrecision *psi0=m_Psi0+(i-m_x1FFT)*m_nyz+(k-m_z1FFT)*4;
	       // thickness of slab is thin, switch loop-j and loop-k for better trip count of simd
#pragma omp simd
	       for (long j=m_slab_z1.ya; j<=m_slab_z1.yb; ++j) {
		  long jtmp1=(j-m_slab_z1.ya)*nz;
		  long jtmp2=(j-m_y1FFT)*m_nz;
		  *(psi0+jtmp2) -= (*(pInc+jtmp1))*Iinc_r-(*(pInc+jtmp1+1))*Iinc_i;
		  *(psi0+jtmp2+1) -= (*(pInc+jtmp1))*Iinc_i+(*(pInc+jtmp1+1))*Iinc_r;
		  *(psi0+jtmp2+2) -= (*(pInc+jtmp1+2))*Iinc_r-(*(pInc+jtmp1+3))*Iinc_i;
		  *(psi0+jtmp2+3) -= (*(pInc+jtmp1+2))*Iinc_i+(*(pInc+jtmp1+3))*Iinc_r;
	       }
	    }
	 }
      }

      // incidence term: total-field/scattered-field at the top transition-layer
      if (m_flgz2Ksi) {
	 long nz=m_slab_z2.nz;
	 long nyz=m_slab_z2.ny*nz;
#pragma omp for collapse(2)
	 for (long i=m_slab_z2.xa; i<=m_slab_z2.xb; ++i) {
	    for (long k=m_slab_z2.za; k<=m_slab_z2.zb; ++k) {
	       QPrecision *pInc=m_Inc_z2+(i-m_slab_z2.xa)*nyz+(k-m_slab_z2.za)*4;
	       QPrecision *psi0=m_Psi0+(i-m_x1FFT)*m_nyz+(k-m_z1FFT)*4;
	       // thickness of slab is thin, switch loop-j and loop-k for better trip count of simd
#pragma omp simd
	       for (long j=m_slab_z2.ya; j<=m_slab_z2.yb; ++j) {
		  long jtmp1=(j-m_slab_z2.ya)*nz;
		  long jtmp2=(j-m_y1FFT)*m_nz;
		  *(psi0+jtmp2) -= (*(pInc+jtmp1))*Iinc_r-(*(pInc+jtmp1+1))*Iinc_i;
		  *(psi0+jtmp2+1) -= (*(pInc+jtmp1))*Iinc_i+(*(pInc+jtmp1+1))*Iinc_r;
		  *(psi0+jtmp2+2) -= (*(pInc+jtmp1+2))*Iinc_r-(*(pInc+jtmp1+3))*Iinc_i;
		  *(psi0+jtmp2+3) -= (*(pInc+jtmp1+2))*Iinc_i+(*(pInc+jtmp1+3))*Iinc_r;
	       }
	    }
	 }
      }

      QPrecision *g1=m_g1-m_i1Core;
      QPrecision *g2=m_g2-m_j1Core;
      QPrecision *g3=m_g3-m_k1Core;
#pragma omp for collapse(2)
      for (long i=m_i1Core; i<=m_i2Core; ++i)
	 for (long j=m_j1Core; j<=m_j2Core; ++j) {
	    QPrecision tmp=(*(g1+i))*(*(g2+j));
	    QPrecision *psi0=m_Psi0+i*m_nyz+j*m_nz;
#pragma omp simd aligned(psi0:CACHE_LINE)
	    for (long k=m_k1Core; k<=m_k2Core; ++k) {
	       QPrecision ttmp=tmp*(*(g3+k));
	       long ktmp=k*4;
	       *(psi0+ktmp) *= ttmp;
	       *(psi0+ktmp+1) *= ttmp;
	       *(psi0+ktmp+2) *= ttmp;
	       *(psi0+ktmp+3) *= ttmp;
	    }
	 }
   } /* ------ end of parallel region ------ */

   MPI_Barrier(m_cartcomm);

   /* swap m_Psi and m_Psi0, the pingpong way of updating Psi */
   QPrecision *tmp_Psi=m_Psi0;
   m_Psi0=m_Psi;
   m_Psi=tmp_Psi;
   ExchangeData();
   ++m_nTau;
}

void CPSTD_MagScat::UpdateX(int myid, long j, long k)
{
   long nyz=m_nyz;
   MKLComplex *p=m_vFFT[myid];
   QPrecision *k2x=m_k2x;
   for (long l_sp=0; l_sp<=2; l_sp+=2) { // up and down spins
      QPrecision *psi=m_Psi+j*m_nz+k*4+l_sp;
      QPrecision *psi0=m_Psi0+j*m_nz+k*4+l_sp;

#pragma omp simd aligned(p:CACHE_LINE)
      for (long i=0; i<m_nXFFT; ++i) {
	 (p+i)->real=*(psi+i*nyz);
	 (p+i)->imag=*(psi+i*nyz+1);
      }

      DftiComputeForward(m_descx[myid],m_vFFT[myid]);
#pragma omp simd aligned(p:CACHE_LINE) aligned(k2x:CACHE_LINE)
      for (long i=0; i<m_nXFFT; ++i) {
	 (p+i)->real *= *(k2x+i);
	 (p+i)->imag *= *(k2x+i);
      }
      DftiComputeBackward(m_descx[myid],m_vFFT[myid]);

      // we only need the non-overlapping part of m_vFFT[myid]
#pragma omp simd aligned(p:CACHE_LINE)
      for (long i=m_i1Core; i<=m_i2Core; ++i) {
	 *(psi0+i*nyz) += m_alpha_2dtau_i*(p+i)->imag;
	 *(psi0+i*nyz+1) -= m_alpha_2dtau_i*(p+i)->real;
      }
   }
}

void CPSTD_MagScat::UpdateY(int myid, long i, long k)
{
   long nz=m_nz;
   MKLComplex *p=m_vFFT[myid];
   QPrecision *k2y=m_k2y;
   for (long l_sp=0; l_sp<=2; l_sp+=2) { // up and down spins
      QPrecision *psi=m_Psi+i*m_nyz+k*4+l_sp;
      QPrecision *psi0=m_Psi0+i*m_nyz+k*4+l_sp;

#pragma omp simd aligned(p:CACHE_LINE)
      for (long j=0; j<m_nYFFT; ++j) {
	 (p+j)->real=*(psi+j*nz);
	 (p+j)->imag=*(psi+j*nz+1);
      }

      DftiComputeForward(m_descy[myid],m_vFFT[myid]);
#pragma omp simd aligned(p:CACHE_LINE) aligned(k2y:CACHE_LINE) 
      for (long j=0; j<m_nYFFT; ++j) {
	 (p+j)->real *= *(k2y+j);
	 (p+j)->imag *= *(k2y+j);
      } 
      DftiComputeBackward(m_descy[myid],m_vFFT[myid]);

      // we only need the non-overlapping part of m_vFFT[myid]
#pragma omp simd aligned(p:CACHE_LINE)
      for (long j=m_j1Core; j<=m_j2Core; ++j) {
	 *(psi0+j*nz) += m_alpha_2dtau_i*(p+j)->imag;
	 *(psi0+j*nz+1) -= m_alpha_2dtau_i*(p+j)->real;
      }
   }
}

void CPSTD_MagScat::UpdateZ(int myid, long i, long j)
{
   MKLComplex *p=m_vFFT[myid];
   QPrecision *k2z=m_k2z;
   for (long l_sp=0; l_sp<=2; l_sp+=2) { // up and down spins
      QPrecision *psi=m_Psi+i*m_nyz+j*m_nz+l_sp;
      QPrecision *psi0=m_Psi0+i*m_nyz+j*m_nz+l_sp;

#pragma omp simd aligned(p:CACHE_LINE)
      for (long k=0; k<m_nZFFT; ++k) {
	 (p+k)->real=*(psi+k*4);
	 (p+k)->imag=*(psi+k*4+1);
      }

      DftiComputeForward(m_descz[myid],m_vFFT[myid]);
#pragma omp simd aligned(p:CACHE_LINE) aligned(k2z:CACHE_LINE)
      for (long k=0; k<m_nZFFT; ++k) {
	 (p+k)->real *= *(k2z+k);
	 (p+k)->imag *= *(k2z+k);
      }
      DftiComputeBackward(m_descz[myid],m_vFFT[myid]);

      // we only need the non-overlapping part of m_vFFT[myid]
#pragma omp simd aligned(p:CACHE_LINE)
      for (long k=m_k1Core; k<=m_k2Core; ++k) {
	 *(psi0+k*4) += m_alpha_2dtau_i*(p+k)->imag;
	 *(psi0+k*4+1) -= m_alpha_2dtau_i*(p+k)->real;
      }
   }
}

void CPSTD_MagScat::ExchangeData()
{
   long lz=m_k1Core*4;

   if (m_nXProcs>1) {
      // send to back and receive from front
      MPI_Sendrecv(m_Psi+m_xaSendOffset,1,m_xTypeExch,m_xaRank,1,
	    m_Psi+m_xbRecvOffset,1,m_xTypeExch,m_xbRank,1,
	    m_cartcomm,&m_status);

      // send to front and receive from back
      MPI_Sendrecv(m_Psi+m_xbSendOffset,1,m_xTypeExch,m_xbRank,2,
	    m_Psi+m_xaRecvOffset,1,m_xTypeExch,m_xaRank,2,
	    m_cartcomm,&m_status);

      // apply weight to the overlapping zone
#pragma omp parallel default(shared) num_threads(m_nThread)
      {
	 // back interface
	 if (m_x1Core>m_x1FFT) { // not the back-surface domain
#pragma omp for
	    for (long j=m_j1Core; j<=m_j2Core; ++j) {
	       for (long k=lz; k<=m_k2Core*4; k+=4) {
		  *(m_Psi+j*m_nz+k) = 0.0;
		  *(m_Psi+j*m_nz+k+1) = 0.0;
		  *(m_Psi+j*m_nz+k+2) = 0.0;
		  *(m_Psi+j*m_nz+k+3) = 0.0;
	       }
	    }
#pragma omp for collapse(2)
	    for (long i=1; i<=NWEIGHT; ++i) {
	       for (long j=m_j1Core; j<=m_j2Core; ++j) {
		  QPrecision tmp=m_weight[i-1];
		  QPrecision *psi=m_Psi+i*m_nyz+j*m_nz;
#pragma omp simd aligned(psi:CACHE_LINE)
		  for (long k=lz; k<=m_k2Core*4; k+=4) {
		     *(psi+k) *= tmp;
		     *(psi+k+1) *= tmp;
		     *(psi+k+2) *= tmp;
		     *(psi+k+3) *= tmp;
		  }
	       }
	    }
	 }

	 // front interface
	 if (m_x2Core<m_x2FFT) { // not the front-surface domain
	    long i2FFT=m_nXFFT-1;
#pragma omp for
	    for (long j=m_j1Core; j<=m_j2Core; ++j) {
	       for (long k=lz; k<=m_k2Core*4; k+=4) {
		  *(m_Psi+i2FFT*m_nyz+j*m_nz+k) = 0.0;
		  *(m_Psi+i2FFT*m_nyz+j*m_nz+k+1) = 0.0;
		  *(m_Psi+i2FFT*m_nyz+j*m_nz+k+2) = 0.0;
		  *(m_Psi+i2FFT*m_nyz+j*m_nz+k+3) = 0.0;
	       }
	    }
#pragma omp for collapse(2)
	    for (long i=i2FFT-NWEIGHT; i<i2FFT; ++i) {
	       for (long j=m_j1Core; j<=m_j2Core; ++j) {
		  QPrecision tmp=m_weight[i2FFT-1-i];
		  QPrecision *psi=m_Psi+i*m_nyz+j*m_nz;
#pragma omp simd aligned(psi:CACHE_LINE)
		  for (long k=lz; k<=m_k2Core*4; k+=4) {
		     *(psi+k) *= tmp;
		     *(psi+k+1) *= tmp;
		     *(psi+k+2) *= tmp;
		     *(psi+k+3) *= tmp;
		  }
	       }
	    }
	 }
      }
   }

   if (m_nYProcs>1) {
      // send to left and receive from right
      MPI_Sendrecv(m_Psi+m_yaSendOffset,1,m_yTypeExch,m_yaRank,3,
	    m_Psi+m_ybRecvOffset,1,m_yTypeExch,m_ybRank,3,
	    m_cartcomm,&m_status);

      // send to right and receive from left
      MPI_Sendrecv(m_Psi+m_ybSendOffset,1,m_yTypeExch,m_ybRank,4,
	    m_Psi+m_yaRecvOffset,1,m_yTypeExch,m_yaRank,4,
	    m_cartcomm,&m_status);

      // apply weight to the overlapping zone
#pragma omp parallel default(shared) num_threads(m_nThread)
      {
	 // left interface
	 if (m_y1Core>m_y1FFT) { // not the left-surface domain
#pragma omp for
	    for (long i=m_i1Core; i<=m_i2Core; ++i) {
	       for (long k=lz; k<=m_k2Core*4; k+=4) {
		  *(m_Psi+i*m_nyz+k) = 0.0;
		  *(m_Psi+i*m_nyz+k+1) = 0.0;
		  *(m_Psi+i*m_nyz+k+2) = 0.0;
		  *(m_Psi+i*m_nyz+k+3) = 0.0;
	       }
	    }
#pragma omp for collapse(2)
	    for (long j=1; j<=NWEIGHT; ++j) {
	       for (long i=m_i1Core; i<=m_i2Core; ++i) {
		  QPrecision tmp=m_weight[j-1];
		  QPrecision *psi=m_Psi+i*m_nyz+j*m_nz;
#pragma omp simd aligned(psi:CACHE_LINE)
		  for (long k=lz; k<=m_k2Core*4; k+=4) {
		     *(psi+k) *= tmp;
		     *(psi+k+1) *= tmp;
		     *(psi+k+2) *= tmp;
		     *(psi+k+3) *= tmp;
		  }
	       }
	    }
	 }

	 // right interface
	 if (m_y2Core<m_y2FFT) { // not the right-surface domain
	    long j2FFT=m_nYFFT-1;
#pragma omp for
	    for (long i=m_i1Core; i<=m_i2Core; ++i) {
	       for (long k=lz; k<=m_k2Core*4; k+=4) {
		  *(m_Psi+i*m_nyz+j2FFT*m_nz+k) = 0.0;
		  *(m_Psi+i*m_nyz+j2FFT*m_nz+k+1) = 0.0;
		  *(m_Psi+i*m_nyz+j2FFT*m_nz+k+2) = 0.0;
		  *(m_Psi+i*m_nyz+j2FFT*m_nz+k+3) = 0.0;
	       }
	    }
#pragma omp for collapse(2)
	    for (long j=j2FFT-NWEIGHT; j<j2FFT; ++j) {
	       for (long i=m_i1Core; i<=m_i2Core; ++i) {
		  QPrecision tmp=m_weight[j2FFT-1-j];
		  QPrecision *psi=m_Psi+i*m_nyz+j*m_nz;
#pragma omp simd aligned(psi:CACHE_LINE)
		  for (long k=lz; k<=m_k2Core*4; k+=4) {
		     *(psi+k) *= tmp;
		     *(psi+k+1) *= tmp;
		     *(psi+k+2) *= tmp;
		     *(psi+k+3) *= tmp;
		  }
	       }
	    }
	 }
      }
   }

   if (m_nZProcs>1) {
      // send to bottom and receive from top
      MPI_Sendrecv(m_Psi+m_zaSendOffset,1,m_zTypeExch,m_zaRank,5,
	    m_Psi+m_zbRecvOffset,1,m_zTypeExch,m_zbRank,5,
	    m_cartcomm,&m_status);

      // send to top and receive from bottom
      MPI_Sendrecv(m_Psi+m_zbSendOffset,1,m_zTypeExch,m_zbRank,6,
	    m_Psi+m_zaRecvOffset,1,m_zTypeExch,m_zaRank,6,
	    m_cartcomm,&m_status);

      // apply weight to the overlapping zone
#pragma omp parallel default(shared) num_threads(m_nThread)
      {
	 // bottom interface
	 if (m_z1Core>m_z1FFT) { // not the bottom-surface domain
#pragma omp for collapse(2)
	    for (long i=m_i1Core; i<=m_i2Core; ++i) {
	       for (long j=m_j1Core; j<=m_j2Core; ++j) {
		  *(m_Psi+i*m_nyz+j*m_nz) = 0.0;
		  *(m_Psi+i*m_nyz+j*m_nz+1) = 0.0;
		  *(m_Psi+i*m_nyz+j*m_nz+2) = 0.0;
		  *(m_Psi+i*m_nyz+j*m_nz+3) = 0.0;
	       }
	    }
#pragma omp for collapse(2)
	    for (long i=m_i1Core; i<=m_i2Core; ++i) {
	       for (long j=m_j1Core; j<=m_j2Core; ++j) {
		  QPrecision *psi=m_Psi+i*m_nyz+j*m_nz;
#pragma omp simd aligned(psi:CACHE_LINE)
		  for (long k=4; k<=NWEIGHT*4; k+=4) {
		     QPrecision tmp=m_weight[k/4-1];
		     *(psi+k) *= tmp;
		     *(psi+k+1) *= tmp;
		     *(psi+k+2) *= tmp;
		     *(psi+k+3) *= tmp;
		  }
	       }
	    }
	 }

	 // top interface
	 if (m_z2Core<m_z2FFT) { // not the top-surface domain
	    long k2FFT=(m_nZFFT-1)*4;
#pragma omp for collapse(2)
	    for (long i=m_i1Core; i<=m_i2Core; ++i) {
	       for (long j=m_j1Core; j<=m_j2Core; ++j) {
		  *(m_Psi+i*m_nyz+j*m_nz+k2FFT) = 0.0;
		  *(m_Psi+i*m_nyz+j*m_nz+k2FFT+1) = 0.0;
		  *(m_Psi+i*m_nyz+j*m_nz+k2FFT+2) = 0.0;
		  *(m_Psi+i*m_nyz+j*m_nz+k2FFT+3) = 0.0;
	       }
	    }
#pragma omp for collapse(2)
	    for (long i=m_i1Core; i<=m_i2Core; ++i) {
	       for (long j=m_j1Core; j<=m_j2Core; ++j) {
		  QPrecision *psi=m_Psi+i*m_nyz+j*m_nz;
#pragma omp simd aligned(psi:CACHE_LINE)
		  for (long k=k2FFT-NWEIGHT*4; k<k2FFT; k+=4) {
		     QPrecision tmp=m_weight[(k2FFT-k)/4-1];
		     *(psi+k) *= tmp;
		     *(psi+k+1) *= tmp;
		     *(psi+k+2) *= tmp;
		     *(psi+k+3) *= tmp;
		  }
	       }
	    }
	 }
      }
   }

   MPI_Barrier(m_cartcomm);
}

long CPSTD_MagScat::IterationsToAccumulateSurfaceTerms()
{
   return (long) (TWOPI/m_dtau+0.5);
}

void CPSTD_MagScat::AccumulateSurfaceTerms()
{
   // incident wave exp(i\vec{k}\cdot\vec{r}-i\omega t)
   // the time-fft factor is thus exp(i\omega t)
   // the time-domain signal of Psi and dPsi is Fourier
   // transformed and registered as the virtual surface
   // data.  Refer to Allen Taflove and Susan C. Hagness,
   // "Computational Electrodynamics, The Finite-Difference
   // Time-domain Method, third Edition", ISBN 1-58053-832-0,
   // page 335, for details of the purpose of this transform.
   QPrecision inc_tau_r=COS(m_nTau*m_dtau);
   QPrecision inc_tau_i=SIN(m_nTau*m_dtau);
   QPrecision inc_tauPS_r=inc_tau_r/2.0;
   QPrecision inc_tauPS_i=inc_tau_i/2.0;

#pragma omp parallel num_threads(m_nThread)
   {
      for (long l_sp=0; l_sp<=2; l_sp+=2) {
	 // 1. Accumulate surface psi
	 // 2. Accumulate $(\hat x,\hat y, \hat z)\cdot\nabla\Psi$ at the surface
	 //  Because the FFT of even length has a defect in computing the 1st order derivative,
	 //  we would calculate the derivatives at an offset of half a grid, then average the
	 //  two adjacent grid values (i.e., linear interpolation).  Another apporach is to
	 //  pad one 0 to the end of the data, thus makes the FFT length odd.  Here, we employ
	 //  the first approach.

	 // the back and front surfaces 
	 if (m_flgx1Vrtl || m_flgx2Vrtl) {
	    QPrecision inc_tauFD_r=inc_tau_r/m_dx;
	    QPrecision inc_tauFD_i=inc_tau_i/m_dx;

#pragma omp for
	    for (long j=m_j1Core; j<=m_j2Core; ++j) {
	       for (long k=m_k1Core; k<=m_k2Core; ++k) {
		  int myid=omp_get_thread_num();
		  MKLComplex *pf=m_vFFT[myid];
		  QPrecision *psi=m_Psi+j*m_nz+k*4+l_sp;
#pragma omp simd aligned(pf:CACHE_LINE)
		  for (long i=0; i<m_nXFFT; ++i) {
		     (pf+i)->real=*(psi+i*m_nyz);
		     (pf+i)->imag=*(psi+i*m_nyz+1);
		  }

		  DftiComputeForward(m_descx[myid],pf);
		  QPrecision *k1x=m_k1x;
#pragma omp simd aligned(pf:CACHE_LINE) aligned(k1x:CACHE_LINE)
		  for (long i=0; i<m_nXFFT; ++i) {
		     QPrecision tmp1_r=(pf+i)->real*(*(k1x+i*2))-(pf+i)->imag*(*(k1x+i*2+1));
		     QPrecision tmp1_i=(pf+i)->real*(*(k1x+i*2+1))+(pf+i)->imag*(*(k1x+i*2));
		     (pf+i)->real=tmp1_r;
		     (pf+i)->imag=tmp1_i;
		  }
		  DftiComputeBackward(m_descx[myid],pf);

		  if (m_flgx1Vrtl) {
		     QPrecision *surfpsix1, *surfdpsiPSx1, *surfdpsiFDx1, *p;
		     surfpsix1=m_SurfPsi_x1+(j-m_j1Core)*m_Surf_x_nz+(k-m_k1Core)*4+l_sp;
		     surfdpsiPSx1=m_SurfDPsi_PS_x1+(j-m_j1Core)*m_Surf_x_nz+(k-m_k1Core)*4+l_sp;
		     surfdpsiFDx1=m_SurfDPsi_FD_x1+(j-m_j1Core)*m_Surf_x_nz+(k-m_k1Core)*4+l_sp;
		     p=psi+(m_X1Vrtl-m_x1FFT)*m_nyz;

		     // 1. Accumulate surface psi
		     // m_X1Vrtl not in x-ABC region
		     *surfpsix1 += (*p)*inc_tau_r-(*(p+1))*inc_tau_i;
		     *(surfpsix1+1) += (*p)*inc_tau_i+(*(p+1))*inc_tau_r;
		     // 2. Accumulate $\frac{\partial\Psi}{\partial x}$ at the surface

		     // the FD approach
		     QPrecision tmp_r =
			  FD4*((*(p+4*m_nyz))-(*(p-4*m_nyz))) + FD3*((*(p+3*m_nyz))-(*(p-3*m_nyz)))
			+ FD2*((*(p+2*m_nyz))-(*(p-2*m_nyz))) + FD1*((*(p+m_nyz))-(*(p-m_nyz)));
		     QPrecision tmp_i =
			  FD4*((*(p+4*m_nyz+1))-(*(p-4*m_nyz+1))) + FD3*((*(p+3*m_nyz+1))-(*(p-3*m_nyz+1)))
			+ FD2*((*(p+2*m_nyz+1))-(*(p-2*m_nyz+1))) + FD1*((*(p+m_nyz+1))-(*(p-m_nyz+1)));

		     *surfdpsiFDx1 += tmp_r*inc_tauFD_r-tmp_i*inc_tauFD_i;
		     *(surfdpsiFDx1+1) += tmp_r*inc_tauFD_i+tmp_i*inc_tauFD_r;

		     // the PS approach
		     MKLComplex *pp=pf+m_X1Vrtl-m_x1FFT;
		     tmp_r=(pp-1)->real+pp->real;
		     tmp_i=(pp-1)->imag+pp->imag;
		     *surfdpsiPSx1 += tmp_r*inc_tauPS_r-tmp_i*inc_tauPS_i;
		     *(surfdpsiPSx1+1) += tmp_r*inc_tauPS_i+tmp_i*inc_tauPS_r;
		  }

		  if (m_flgx2Vrtl) {
		     QPrecision *surfpsix2, *surfdpsiPSx2, *surfdpsiFDx2, *p;
		     surfpsix2=m_SurfPsi_x2+(j-m_j1Core)*m_Surf_x_nz+(k-m_k1Core)*4+l_sp;
		     surfdpsiPSx2=m_SurfDPsi_PS_x2+(j-m_j1Core)*m_Surf_x_nz+(k-m_k1Core)*4+l_sp;
		     surfdpsiFDx2=m_SurfDPsi_FD_x2+(j-m_j1Core)*m_Surf_x_nz+(k-m_k1Core)*4+l_sp;
		     p=psi+(m_X2Vrtl-m_x1FFT)*m_nyz;

		     // 1. Accumulate surface psi
		     // m_X2Vrtl not in x-ABC region
		     *surfpsix2 += (*p)*inc_tau_r-(*(p+1))*inc_tau_i;
		     *(surfpsix2+1) += (*p)*inc_tau_i+(*(p+1))*inc_tau_r;
		     // 2. Accumulate $\frac{\partial\Psi}{\partial x}$ at the surface
		     
		     // the FD approach
		     QPrecision tmp_r =
			  FD4*((*(p+4*m_nyz))-(*(p-4*m_nyz))) + FD3*((*(p+3*m_nyz))-(*(p-3*m_nyz)))
			+ FD2*((*(p+2*m_nyz))-(*(p-2*m_nyz))) + FD1*((*(p+m_nyz))-(*(p-m_nyz)));
		     QPrecision tmp_i =
			  FD4*((*(p+4*m_nyz+1))-(*(p-4*m_nyz+1))) + FD3*((*(p+3*m_nyz+1))-(*(p-3*m_nyz+1)))
			+ FD2*((*(p+2*m_nyz+1))-(*(p-2*m_nyz+1))) + FD1*((*(p+m_nyz+1))-(*(p-m_nyz+1)));

		     *surfdpsiFDx2 += tmp_r*inc_tauFD_r-tmp_i*inc_tauFD_i;
		     *(surfdpsiFDx2+1) += tmp_r*inc_tauFD_i+tmp_i*inc_tauFD_r;

		     // the PS approach
		     MKLComplex *pp=pf+m_X2Vrtl-m_x1FFT;
		     tmp_r=(pp-1)->real+pp->real;
		     tmp_i=(pp-1)->imag+pp->imag;
		     *surfdpsiPSx2 += tmp_r*inc_tauPS_r-tmp_i*inc_tauPS_i;
		     *(surfdpsiPSx2+1) += tmp_r*inc_tauPS_i+tmp_i*inc_tauPS_r;
		  }
	       }
	    }
	 }

	 // the left and right surfaces 
	 if (m_flgy1Vrtl || m_flgy2Vrtl) {
	    QPrecision inc_tauFD_r=inc_tau_r/m_dy;
	    QPrecision inc_tauFD_i=inc_tau_i/m_dy;

#pragma omp for
	    for (long i=m_i1Core; i<=m_i2Core; ++i) {
	       for (long k=m_k1Core; k<=m_k2Core; ++k) {
		  int myid=omp_get_thread_num();
		  MKLComplex *pf=m_vFFT[myid];
		  QPrecision *psi=m_Psi+i*m_nyz+k*4+l_sp;
#pragma omp simd aligned(pf:CACHE_LINE)
		  for (long j=0; j<m_nYFFT; ++j) {
		     (pf+j)->real=*(psi+j*m_nz);
		     (pf+j)->imag=*(psi+j*m_nz+1);
		  }

		  DftiComputeForward(m_descy[myid],pf);
		  QPrecision *k1y=m_k1y;
#pragma omp simd aligned(pf:CACHE_LINE) aligned(k1y:CACHE_LINE)
		  for (long j=0; j<m_nYFFT; ++j) {
		     QPrecision tmp1_r=(pf+j)->real*(*(k1y+j*2))-(pf+j)->imag*(*(k1y+j*2+1));
		     QPrecision tmp1_i=(pf+j)->real*(*(k1y+j*2+1))+(pf+j)->imag*(*(k1y+j*2));
		     (pf+j)->real=tmp1_r;
		     (pf+j)->imag=tmp1_i;
		  }
		  DftiComputeBackward(m_descy[myid],pf);

		  if (m_flgy1Vrtl) {
		     QPrecision *surfpsiy1, *surfdpsiPSy1, *surfdpsiFDy1, *p;
		     surfpsiy1=m_SurfPsi_y1+(i-m_i1Core)*m_Surf_y_nz+(k-m_k1Core)*4+l_sp;
		     surfdpsiPSy1=m_SurfDPsi_PS_y1+(i-m_i1Core)*m_Surf_y_nz+(k-m_k1Core)*4+l_sp;
		     surfdpsiFDy1=m_SurfDPsi_FD_y1+(i-m_i1Core)*m_Surf_y_nz+(k-m_k1Core)*4+l_sp;
		     p=psi+(m_Y1Vrtl-m_y1FFT)*m_nz;

		     // 1. Accumulate surface psi
		     // m_Y1Vrtl not in y-ABC region
		     *surfpsiy1 += (*p)*inc_tau_r-(*(p+1))*inc_tau_i;
		     *(surfpsiy1+1) += (*p)*inc_tau_i+(*(p+1))*inc_tau_r;
		     // 2. Accumulate $\frac{\partial\Psi}{\partial y}$ at the surface
		     
		     // the FD approach
		     QPrecision tmp_r =
			  FD4*((*(p+4*m_nz))-(*(p-4*m_nz))) + FD3*((*(p+3*m_nz))-(*(p-3*m_nz)))
			+ FD2*((*(p+2*m_nz))-(*(p-2*m_nz))) + FD1*((*(p+m_nz))-(*(p-m_nz)));
		     QPrecision tmp_i =
			  FD4*((*(p+4*m_nz+1))-(*(p-4*m_nz+1))) + FD3*((*(p+3*m_nz+1))-(*(p-3*m_nz+1)))
			+ FD2*((*(p+2*m_nz+1))-(*(p-2*m_nz+1))) + FD1*((*(p+m_nz+1))-(*(p-m_nz+1)));

		     *surfdpsiFDy1 += tmp_r*inc_tauFD_r-tmp_i*inc_tauFD_i;
		     *(surfdpsiFDy1+1) += tmp_r*inc_tauFD_i+tmp_i*inc_tauFD_r;

		     // the PS approach
		     MKLComplex *pp=pf+m_Y1Vrtl-m_y1FFT;
		     tmp_r=(pp-1)->real+pp->real;
		     tmp_i=(pp-1)->imag+pp->imag;
		     *surfdpsiPSy1 += tmp_r*inc_tauPS_r-tmp_i*inc_tauPS_i;
		     *(surfdpsiPSy1+1) += tmp_r*inc_tauPS_i+tmp_i*inc_tauPS_r;
		  }

		  if (m_flgy2Vrtl) {
		     QPrecision *surfpsiy2, *surfdpsiPSy2, *surfdpsiFDy2, *p;
		     surfpsiy2=m_SurfPsi_y2+(i-m_i1Core)*m_Surf_y_nz+(k-m_k1Core)*4+l_sp;
		     surfdpsiPSy2=m_SurfDPsi_PS_y2+(i-m_i1Core)*m_Surf_y_nz+(k-m_k1Core)*4+l_sp;
		     surfdpsiFDy2=m_SurfDPsi_FD_y2+(i-m_i1Core)*m_Surf_y_nz+(k-m_k1Core)*4+l_sp;
		     p=psi+(m_Y2Vrtl-m_y1FFT)*m_nz;

		     // 1. Accumulate surface psi
		     // m_Y2Vrtl not in y-ABC region
		     *surfpsiy2 += (*p)*inc_tau_r-(*(p+1))*inc_tau_i;;
		     *(surfpsiy2+1) += (*p)*inc_tau_i+(*(p+1))*inc_tau_r;;
		     // 2. Accumulate $\frac{\partial\Psi}{\partial y}$ at the surface
		     
		     // the FD approach
		     QPrecision tmp_r =
			  FD4*((*(p+4*m_nz))-(*(p-4*m_nz))) + FD3*((*(p+3*m_nz))-(*(p-3*m_nz)))
			+ FD2*((*(p+2*m_nz))-(*(p-2*m_nz))) + FD1*((*(p+m_nz))-(*(p-m_nz)));
		     QPrecision tmp_i =
			  FD4*((*(p+4*m_nz+1))-(*(p-4*m_nz+1))) + FD3*((*(p+3*m_nz+1))-(*(p-3*m_nz+1)))
			+ FD2*((*(p+2*m_nz+1))-(*(p-2*m_nz+1))) + FD1*((*(p+m_nz+1))-(*(p-m_nz+1)));

		     *surfdpsiFDy2 += tmp_r*inc_tauFD_r-tmp_i*inc_tauFD_i;
		     *(surfdpsiFDy2+1) += tmp_r*inc_tauFD_i+tmp_i*inc_tauFD_r;

		     // the PS approach
		     MKLComplex *pp=pf+m_Y2Vrtl-m_y1FFT;
		     tmp_r=(pp-1)->real+pp->real;
		     tmp_i=(pp-1)->imag+pp->imag;
		     *surfdpsiPSy2 += tmp_r*inc_tauPS_r-tmp_i*inc_tauPS_i;
		     *(surfdpsiPSy2+1) += tmp_r*inc_tauPS_i+tmp_i*inc_tauPS_r;
		  }
	       }
	    }
	 }

	 // the bottom and top surfaces 
	 if (m_flgz1Vrtl || m_flgz2Vrtl) {
	    QPrecision inc_tauFD_r=inc_tau_r/m_dz;
	    QPrecision inc_tauFD_i=inc_tau_i/m_dz;

#pragma omp for
	    for (long i=m_i1Core; i<=m_i2Core; ++i) {
	       for (long j=m_j1Core; j<=m_j2Core; ++j) {
		  int myid=omp_get_thread_num();
		  MKLComplex *pf=m_vFFT[myid];
		  QPrecision *psi=m_Psi+i*m_nyz+j*m_nz+l_sp;
#pragma omp simd aligned(pf:CACHE_LINE)
		  for (long k=0; k<m_nZFFT; ++k) {
		     (pf+k)->real=*(psi+k*4);
		     (pf+k)->imag=*(psi+k*4+1);
		  }

		  DftiComputeForward(m_descz[myid],pf);
		  QPrecision *k1z=m_k1z;
#pragma omp simd aligned(pf:CACHE_LINE) aligned(k1z:CACHE_LINE)
		  for (long k=0; k<m_nZFFT; ++k) {
		     QPrecision tmp1_r=(pf+k)->real*(*(k1z+k*2))-(pf+k)->imag*(*(k1z+k*2+1));
		     QPrecision tmp1_i=(pf+k)->real*(*(k1z+k*2+1))+(pf+k)->imag*(*(k1z+k*2));
		     (pf+k)->real=tmp1_r;
		     (pf+k)->imag=tmp1_i;
		  }

		  DftiComputeBackward(m_descz[myid],pf);

		  if (m_flgz1Vrtl) {
		     QPrecision *surfpsiz1, *surfdpsiPSz1, *surfdpsiFDz1, *p;
		     surfpsiz1=m_SurfPsi_z1+(i-m_i1Core)*m_Surf_z_ny+(j-m_j1Core)*4+l_sp;
		     surfdpsiPSz1=m_SurfDPsi_PS_z1+(i-m_i1Core)*m_Surf_z_ny+(j-m_j1Core)*4+l_sp;
		     surfdpsiFDz1=m_SurfDPsi_FD_z1+(i-m_i1Core)*m_Surf_z_ny+(j-m_j1Core)*4+l_sp;
		     p=psi+(m_Z1Vrtl-m_z1FFT)*4;

		     // 1. accumulate surface psi
		     // m_Z1Vrtl not in z-ABC region
		     *surfpsiz1 += (*p)*inc_tau_r-(*(p+1))*inc_tau_i;
		     *(surfpsiz1+1) += (*p)*inc_tau_i+(*(p+1))*inc_tau_r;
		     // 2. Accumulate $\frac{\partial\Psi}{\partial z}$ at the surface
		     
		     // the FD approach
		     QPrecision tmp_r =
			  FD4*((*(p+16))-(*(p-16))) + FD3*((*(p+12))-(*(p-12)))
			+ FD2*((*(p+8))-(*(p-8))) + FD1*((*(p+4))-(*(p-4)));
		     QPrecision tmp_i =
			  FD4*((*(p+17))-(*(p-15))) + FD3*((*(p+13))-(*(p-11)))
			+ FD2*((*(p+9))-(*(p-7))) + FD1*((*(p+5))-(*(p-3)));

		     *surfdpsiFDz1 += tmp_r*inc_tauFD_r-tmp_i*inc_tauFD_i;
		     *(surfdpsiFDz1+1) += tmp_r*inc_tauFD_i+tmp_i*inc_tauFD_r;

		     // the PS approach
		     MKLComplex *pp=pf+m_Z1Vrtl-m_z1FFT;
		     tmp_r=(pp-1)->real+pp->real;
		     tmp_i=(pp-1)->imag+pp->imag;
		     *surfdpsiPSz1 += tmp_r*inc_tauPS_r-tmp_i*inc_tauPS_i;
		     *(surfdpsiPSz1+1) += tmp_r*inc_tauPS_i+tmp_i*inc_tauPS_r;
		  }

		  if (m_flgz2Vrtl) {
		     QPrecision *surfpsiz2, *surfdpsiPSz2, *surfdpsiFDz2, *p;
		     surfpsiz2=m_SurfPsi_z2+(i-m_i1Core)*m_Surf_z_ny+(j-m_j1Core)*4+l_sp;
		     surfdpsiPSz2=m_SurfDPsi_PS_z2+(i-m_i1Core)*m_Surf_z_ny+(j-m_j1Core)*4+l_sp;
		     surfdpsiFDz2=m_SurfDPsi_FD_z2+(i-m_i1Core)*m_Surf_z_ny+(j-m_j1Core)*4+l_sp;
		     p=psi+(m_Z2Vrtl-m_z1FFT)*4;

		     // 1. accumulate surface psi
		     // m_Z2Vrtl not in z-ABC region
		     *surfpsiz2 += (*p)*inc_tau_r-(*(p+1))*inc_tau_i;
		     *(surfpsiz2+1) += (*p)*inc_tau_i+(*(p+1))*inc_tau_r;
		     // 2. Accumulate $\frac{\partial\Psi}{\partial z}$ at the surface
		     
		     // the FD approach
		     QPrecision tmp_r =
			  FD4*((*(p+16))-(*(p-16))) + FD3*((*(p+12))-(*(p-12)))
			+ FD2*((*(p+8))-(*(p-8))) + FD1*((*(p+4))-(*(p-4)));
		     QPrecision tmp_i =
			  FD4*((*(p+17))-(*(p-15))) + FD3*((*(p+13))-(*(p-11)))
			+ FD2*((*(p+9))-(*(p-7))) + FD1*((*(p+5))-(*(p-3)));

		     *surfdpsiFDz2 += tmp_r*inc_tauFD_r-tmp_i*inc_tauFD_i;
		     *(surfdpsiFDz2+1) += tmp_r*inc_tauFD_i+tmp_i*inc_tauFD_r;

		     // the PS approach
		     MKLComplex *pp=pf+m_Z2Vrtl-m_z1FFT;
		     tmp_r=(pp-1)->real+pp->real;
		     tmp_i=(pp-1)->imag+pp->imag;
		     *surfdpsiPSz2 += tmp_r*inc_tauPS_r-tmp_i*inc_tauPS_i;
		     *(surfdpsiPSz2+1) += tmp_r*inc_tauPS_i+tmp_i*inc_tauPS_r;
		  }
	       }
	    }
	 }
      }
   } /* ------ end of parallel region ------ */

   m_nSur++;
   MPI_Barrier(m_cartcomm);
}

void CPSTD_MagScat::FinalizeSurfaceTerms()
{
   // back surface data
   if (m_flgx1Vrtl) {
#pragma omp parallel for num_threads(m_nThread)
      for (long j=0; j<m_Surf_x_ny; ++j) {
	 QPrecision *psurf=m_SurfPsi_x1+j*m_Surf_x_nz;
	 QPrecision *psurfd_PS=m_SurfDPsi_PS_x1+j*m_Surf_x_nz;
	 QPrecision *psurfd_FD=m_SurfDPsi_FD_x1+j*m_Surf_x_nz;
#pragma omp simd aligned(psurf:CACHE_LINE) aligned(psurfd_PS:CACHE_LINE) aligned(psurfd_FD:CACHE_LINE)
         for (long k=0; k<m_Surf_x_nz; ++k) {
            // We only employed incident plane wave, so the accumulated incident
            // wave factor (each accumulation gives exp[-i\omega t]*exp[i\omega t])
            // is just m_nSur
            *(psurf+k) /= m_nSur;
            *(psurfd_PS+k) /= m_nSur;
            *(psurfd_FD+k) /= m_nSur;
         }
      }
   }

   // front surface data
   if (m_flgx2Vrtl) {
#pragma omp parallel for num_threads(m_nThread)
      for (long j=0; j<m_Surf_x_ny; ++j) {
	 QPrecision *psurf=m_SurfPsi_x2+j*m_Surf_x_nz;
	 QPrecision *psurfd_PS=m_SurfDPsi_PS_x2+j*m_Surf_x_nz;
	 QPrecision *psurfd_FD=m_SurfDPsi_FD_x2+j*m_Surf_x_nz;
#pragma omp simd aligned(psurf:CACHE_LINE) aligned(psurfd_PS:CACHE_LINE) aligned(psurfd_FD:CACHE_LINE)
         for (long k=0; k<m_Surf_x_nz; ++k) {
            *(psurf+k) /= m_nSur;
            *(psurfd_PS+k) /= m_nSur;
            *(psurfd_FD+k) /= m_nSur;
         }
      }
   }

   // left surface data
   if (m_flgy1Vrtl) {
#pragma omp parallel for num_threads(m_nThread)
      for (long i=0; i<m_Surf_y_nx; ++i) {
	 QPrecision *psurf=m_SurfPsi_y1+i*m_Surf_y_nz;
	 QPrecision *psurfd_PS=m_SurfDPsi_PS_y1+i*m_Surf_y_nz;
	 QPrecision *psurfd_FD=m_SurfDPsi_FD_y1+i*m_Surf_y_nz;
#pragma omp simd aligned(psurf:CACHE_LINE) aligned(psurfd_PS:CACHE_LINE) aligned(psurfd_FD:CACHE_LINE)
         for (long k=0; k<m_Surf_y_nz; ++k) {
            *(psurf+k) /= m_nSur;
            *(psurfd_PS+k) /= m_nSur;
            *(psurfd_FD+k) /= m_nSur;
         }
      }
   }

   // right surface data
   if (m_flgy2Vrtl) {
#pragma omp parallel for num_threads(m_nThread)
      for (long i=0; i<m_Surf_y_nx; ++i) {
	 QPrecision *psurf=m_SurfPsi_y2+i*m_Surf_y_nz;
	 QPrecision *psurfd_PS=m_SurfDPsi_PS_y2+i*m_Surf_y_nz;
	 QPrecision *psurfd_FD=m_SurfDPsi_FD_y2+i*m_Surf_y_nz;
#pragma omp simd aligned(psurf:CACHE_LINE) aligned(psurfd_PS:CACHE_LINE) aligned(psurfd_FD:CACHE_LINE)
         for (long k=0; k<m_Surf_y_nz; ++k) {
            *(psurf+k) /= m_nSur;
            *(psurfd_PS+k) /= m_nSur;
            *(psurfd_FD+k) /= m_nSur;
         }
      }
   }

   // bottom surface data
   if (m_flgz1Vrtl) {
#pragma omp parallel for num_threads(m_nThread)
      for (long i=0; i<m_Surf_z_nx; ++i) {
	 QPrecision *psurf=m_SurfPsi_z1+i*m_Surf_z_ny;
	 QPrecision *psurfd_PS=m_SurfDPsi_PS_z1+i*m_Surf_z_ny;
	 QPrecision *psurfd_FD=m_SurfDPsi_FD_z1+i*m_Surf_z_ny;
#pragma omp simd aligned(psurf:CACHE_LINE) aligned(psurfd_PS:CACHE_LINE) aligned(psurfd_FD:CACHE_LINE)
         for (long j=0; j<m_Surf_z_ny; ++j) {
            *(psurf+j) /= m_nSur;
            *(psurfd_PS+j) /= m_nSur;
            *(psurfd_FD+j) /= m_nSur;
         }
      }
   }

   // top surface data
   if (m_flgz2Vrtl) {
#pragma omp parallel for num_threads(m_nThread)
      for (long i=0; i<m_Surf_z_nx; ++i) {
	 QPrecision *psurf=m_SurfPsi_z2+i*m_Surf_z_ny;
	 QPrecision *psurfd_PS=m_SurfDPsi_PS_z2+i*m_Surf_z_ny;
	 QPrecision *psurfd_FD=m_SurfDPsi_FD_z2+i*m_Surf_z_ny;
#pragma omp simd aligned(psurf:CACHE_LINE) aligned(psurfd_PS:CACHE_LINE) aligned(psurfd_FD:CACHE_LINE)
         for (long j=0; j<m_Surf_z_ny; ++j) {
            *(psurf+j) /= m_nSur;
            *(psurfd_PS+j) /= m_nSur;
            *(psurfd_FD+j) /= m_nSur;
         }
      }
   }
}

void CPSTD_MagScat::ResetSurfaceTerms()
{
   m_nSur=0;
   if (m_flgx1Vrtl) {
      memset((void *) m_SurfPsi_x1, 0, m_Surf_x_ny*m_Surf_x_nz*sizeof(QPrecision));
      memset((void *) m_SurfDPsi_PS_x1, 0, m_Surf_x_ny*m_Surf_x_nz*sizeof(QPrecision));
      memset((void *) m_SurfDPsi_FD_x1, 0, m_Surf_x_ny*m_Surf_x_nz*sizeof(QPrecision));
   }
   if (m_flgx2Vrtl) {
      memset((void *) m_SurfPsi_x2, 0, m_Surf_x_ny*m_Surf_x_nz*sizeof(QPrecision));
      memset((void *) m_SurfDPsi_PS_x2, 0, m_Surf_x_ny*m_Surf_x_nz*sizeof(QPrecision));
      memset((void *) m_SurfDPsi_FD_x2, 0, m_Surf_x_ny*m_Surf_x_nz*sizeof(QPrecision));
   }
   if (m_flgy1Vrtl) {
      memset((void *) m_SurfPsi_y1, 0, m_Surf_y_nx*m_Surf_y_nz*sizeof(QPrecision));
      memset((void *) m_SurfDPsi_PS_y1, 0, m_Surf_y_nx*m_Surf_y_nz*sizeof(QPrecision));
      memset((void *) m_SurfDPsi_FD_y1, 0, m_Surf_y_nx*m_Surf_y_nz*sizeof(QPrecision));
   }
   if (m_flgy2Vrtl) {
      memset((void *) m_SurfPsi_y2, 0, m_Surf_y_nx*m_Surf_y_nz*sizeof(QPrecision));
      memset((void *) m_SurfDPsi_PS_y2, 0, m_Surf_y_nx*m_Surf_y_nz*sizeof(QPrecision));
      memset((void *) m_SurfDPsi_FD_y2, 0, m_Surf_y_nx*m_Surf_y_nz*sizeof(QPrecision));
   }
   if (m_flgz1Vrtl) {
      memset((void *) m_SurfPsi_z1, 0, m_Surf_z_nx*m_Surf_z_ny*sizeof(QPrecision));
      memset((void *) m_SurfDPsi_PS_z1, 0, m_Surf_z_nx*m_Surf_z_ny*sizeof(QPrecision));
      memset((void *) m_SurfDPsi_FD_z1, 0, m_Surf_z_nx*m_Surf_z_ny*sizeof(QPrecision));
   }
   if (m_flgz2Vrtl) {
      memset((void *) m_SurfPsi_z2, 0, m_Surf_z_nx*m_Surf_z_ny*sizeof(QPrecision));
      memset((void *) m_SurfDPsi_PS_z2, 0, m_Surf_z_nx*m_Surf_z_ny*sizeof(QPrecision));
      memset((void *) m_SurfDPsi_FD_z2, 0, m_Surf_z_nx*m_Surf_z_ny*sizeof(QPrecision));
   }
}

void CPSTD_MagScat::Save_g()
{
   stringstream sstr;

   if (!m_g1) {
      sstr << "Rank " << m_Rank << ": m_g1 not initilized";
      cerr << sstr.str() << endl;
   } else {
      sstr << "Rank" << m_Rank << "_g1.dat";
      ofstream ofs(sstr.str().c_str(), ios::out);
      if (!ofs) {
	 sstr << " failed";
	 cerr << sstr.str() << endl;
      } else {
	 for (long i=0; i<m_nXFFT; ++i) {
	    ofs << resetiosflags(ios::scientific) << i << "\t" << \
	       setiosflags(ios::scientific) << setprecision(16) \
	       << m_g1[i] << endl;
	 }
	 ofs.close();
      }
   }

   sstr.clear(); sstr.str(string());
   if (!m_g2) {
      sstr << "Rank " << m_Rank << ": m_g2 not initilized";
      cerr << sstr.str() << endl;
   } else {
      sstr << "Rank" << m_Rank << "_g2.dat";
      ofstream ofs(sstr.str().c_str(), ios::out);
      if (!ofs) {
	 sstr << " failed";
	 cerr << sstr.str() << endl;
      } else {
	 for (long j=0; j<m_nYFFT; ++j) {
	    ofs << resetiosflags(ios::scientific) << j << "\t" << \
	       setiosflags(ios::scientific) << setprecision(16) \
	       << m_g2[j] << endl;
	 }
	 ofs.close();
      }
   }

   sstr.clear(); sstr.str(string());
   if (!m_g3) {
      sstr << "Rank " << m_Rank << ": m_g3 not initilized";
      cerr << sstr.str() << endl;
   } else {
      sstr << "Rank" << m_Rank << "_g3.dat";
      ofstream ofs(sstr.str().c_str(), ios::out);
      if (!ofs) {
	 sstr << " failed";
	 cerr << sstr.str() << endl;
      } else {
	 for (long k=0; k<m_nZFFT; ++k) {
	    ofs << resetiosflags(ios::scientific) << k << "\t" << \
	       setiosflags(ios::scientific) << setprecision(16) \
	       << m_g3[k] << endl;
	 }
	 ofs.close();
      }
   }
}

void CPSTD_MagScat::SavePsi_XPlane(string file_name_prefix, QPrecision x0)
{
   long i0=(long) (x0/m_dx+m_OrigX);
   SavePsi_XPlane(file_name_prefix, i0);
}

void CPSTD_MagScat::SavePsi_XPlane(string file_name_prefix, long i0)
{
   SavePsi_XPlane(file_name_prefix, 0, i0);
   SavePsi_XPlane(file_name_prefix, 2, i0);
}

void CPSTD_MagScat::SavePsi_XPlane(string file_name_prefix, long l_sp, long i0)
{
   if (i0<m_x1Core || i0>m_x2Core) return;
   stringstream sstr;

   // real part
   sstr << file_name_prefix << m_nTau << "_Rank" << m_Rank << "_rx" << i0;
   switch (l_sp) {
      case 0:
	 sstr << "_u.dat";
	 break;
      case 2:
	 sstr << "_d.dat";
	 break;
      default:
	 return;
   }

   ofstream ofsr(sstr.str().c_str(), ios::out);
   if (!ofsr) {
      sstr << " failed";
      cerr << sstr.str() << endl;
   } else {
      for (long j=0; j<m_nYFFT; ++j) {
	 for (long k=0; k<m_nZFFT; ++k) 
	    ofsr << setiosflags(ios::scientific) << setprecision(16) \
	       << *(m_Psi+(i0-m_x1FFT)*m_nyz+j*m_nz+k*4+l_sp) << "\t";
	 ofsr << endl;
      }
      ofsr.close();
   }

   // imaginary part
   sstr.clear(); sstr.str(string());
   sstr << file_name_prefix << m_nTau << "_Rank" << m_Rank << "_ix" << i0;
   switch (l_sp) {
      case 0:
	 sstr << "_u.dat";
	 break;
      case 2:
	 sstr << "_d.dat";
	 break;
      default:
	 return;
   }
   ofstream ofsi(sstr.str().c_str(), ios::out);
   if (!ofsi) {
      sstr << " failed";
      cerr << sstr.str() << endl;
   } else {
      for (long j=0; j<m_nYFFT; ++j) {
	 for (long k=0; k<m_nZFFT; ++k) 
	    ofsi << setiosflags(ios::scientific) << setprecision(16) \
	       << *(m_Psi+(i0-m_x1FFT)*m_nyz+j*m_nz+k*4+1+l_sp) << "\t";
	 ofsi << endl;
      }
      ofsi.close();
   }
}

void CPSTD_MagScat::SavePsi_YPlane(string file_name_prefix, QPrecision y0)
{
   long j0=(long) (y0/m_dy+m_OrigY);
   SavePsi_YPlane(file_name_prefix, j0);
}

void CPSTD_MagScat::SavePsi_YPlane(string file_name_prefix, long j0)
{
   SavePsi_YPlane(file_name_prefix, 0, j0);
   SavePsi_YPlane(file_name_prefix, 2, j0);
}

void CPSTD_MagScat::SavePsi_YPlane(string file_name_prefix, long l_sp, long j0)
{
   if (j0<m_y1Core || j0>m_y2Core) return;
   stringstream sstr;

   // real part
   sstr << file_name_prefix << m_nTau << "_Rank" << m_Rank << "_ry" << j0;
   switch (l_sp) {
      case 0:
	 sstr << "_u.dat";
	 break;
      case 2:
	 sstr << "_d.dat";
	 break;
      default:
	 return;
   }
   ofstream ofsr(sstr.str().c_str(), ios::out);
   if (!ofsr) {
      sstr << " failed";
      cerr << sstr.str() << endl;
   } else {
      for (long i=0; i<m_nXFFT; ++i) {
	 for (long k=0; k<m_nZFFT; ++k) 
	    ofsr << setiosflags(ios::scientific) << setprecision(16) \
	       << *(m_Psi+i*m_nyz+(j0-m_y1FFT)*m_nz+k*4+l_sp) << "\t";
	 ofsr << endl;
      }
      ofsr.close();
   }

   // imaginary part
   sstr.clear(); sstr.str(string());
   sstr << file_name_prefix << m_nTau << "_Rank" << m_Rank << "_iy" << j0;
   switch (l_sp) {
      case 0:
	 sstr << "_u.dat";
	 break;
      case 2:
	 sstr << "_d.dat";
	 break;
      default:
	 return;
   }
   ofstream ofsi(sstr.str().c_str(), ios::out);
   if (!ofsi) {
      sstr << " failed";
      cerr << sstr.str() << endl;
   } else {
      for (long i=0; i<m_nXFFT; ++i) {
	 for (long k=0; k<m_nZFFT; ++k) 
	    ofsi << setiosflags(ios::scientific) << setprecision(16) \
	       << *(m_Psi+i*m_nyz+(j0-m_y1FFT)*m_nz+k*4+1+l_sp) << "\t";
	 ofsi << endl;
      }
      ofsi.close();
   }
}

void CPSTD_MagScat::SavePsi_ZPlane(string file_name_prefix, QPrecision z0)
{
   long k0=(long) (z0/m_dz+m_OrigZ);
   SavePsi_ZPlane(file_name_prefix, k0);
}

void CPSTD_MagScat::SavePsi_ZPlane(string file_name_prefix, long k0)
{
   SavePsi_ZPlane(file_name_prefix, 0, k0);
   SavePsi_ZPlane(file_name_prefix, 2, k0);
}

void CPSTD_MagScat::SavePsi_ZPlane(string file_name_prefix, long l_sp, long k0)
{
   if (k0<m_z1Core || k0>m_z2Core) return;
   stringstream sstr;

   // real part
   sstr << file_name_prefix << m_nTau << "_Rank" << m_Rank << "_rz" << k0;
   switch (l_sp) {
      case 0:
	 sstr << "_u.dat";
	 break;
      case 2:
	 sstr << "_d.dat";
	 break;
      default:
	 return;
   }
   ofstream ofsr(sstr.str().c_str(), ios::out);
   if (!ofsr) {
      sstr << " failed";
      cerr << sstr.str() << endl;
   } else {
      for (long i=0; i<m_nXFFT; ++i) {
	 for (long j=0; j<m_nYFFT; ++j) 
	    ofsr << setiosflags(ios::scientific) << setprecision(16) \
	       << *(m_Psi+i*m_nyz+j*m_nz+(k0-m_z1FFT)*4+l_sp) << "\t";
	 ofsr << endl;
      }
      ofsr.close();
   }

   // imaginary part
   sstr.clear(); sstr.str(string());
   sstr << file_name_prefix << m_nTau << "_Rank" << m_Rank << "_iz" << k0;
   switch (l_sp) {
      case 0:
	 sstr << "_u.dat";
	 break;
      case 2:
	 sstr << "_d.dat";
	 break;
      default:
	 return;
   }
   ofstream ofsi(sstr.str().c_str(), ios::out);
   if (!ofsi) {
      sstr << " failed";
      cerr << sstr.str() << endl;
   } else {
      for (long i=0; i<m_nXFFT; ++i) {
	 for (long j=0; j<m_nYFFT; ++j) 
	    ofsi << setiosflags(ios::scientific) << setprecision(16) \
	       << *(m_Psi+i*m_nyz+j*m_nz+(k0-m_z1FFT)*4+1+l_sp) << "\t";
	 ofsi << endl;
      }
      ofsi.close();
   }
}

void CPSTD_MagScat::SaveVirtualSurfaces(string file_name_prefix)
{
   if(!m_nSur) {
      if (!m_Rank)
	 cerr << "Warning: Virtual Surface terms empty.  No file to be saved." << endl;
      return;
   }

   FinalizeSurfaceTerms();

   SaveSurfaceTerms(file_name_prefix, 0, PS);	// DPsi's along surface normals calculated using PS or FD are saved separately.
   SaveSurfaceTerms(file_name_prefix, 0, FD);	// Surface Psi's will be saved twice to facilitate post-processing.
   SaveSurfaceTerms(file_name_prefix, 2, PS);	// up and down spin components are saved seperately, and
   SaveSurfaceTerms(file_name_prefix, 2, FD);	// near-to-far up and down components can be processed independently.

   ResetSurfaceTerms();
}

void CPSTD_MagScat::SaveSurfaceTerms(string file_name_prefix, long l_sp, DerivativeType PSvsFD)
{
   QPrecision *pDPsi_x1, *pDPsi_x2, *pDPsi_y1, *pDPsi_y2, *pDPsi_z1, *pDPsi_z2;
   stringstream sstr;
   switch (PSvsFD) {
      case PS:
	 sstr << file_name_prefix.c_str() << "_PS_" << m_nTau-m_nSur;	// m_nTau-m_nSur is the starting collection 
	 								// time index of these surface terms
	 pDPsi_x1=m_SurfDPsi_PS_x1; pDPsi_x2=m_SurfDPsi_PS_x2;
	 pDPsi_y1=m_SurfDPsi_PS_y1; pDPsi_y2=m_SurfDPsi_PS_y2;
	 pDPsi_z1=m_SurfDPsi_PS_z1; pDPsi_z2=m_SurfDPsi_PS_z2;
	 break;
      case FD:
	 sstr << file_name_prefix.c_str() << "_FD_" << m_nTau-m_nSur;
	 pDPsi_x1=m_SurfDPsi_FD_x1; pDPsi_x2=m_SurfDPsi_FD_x2;
	 pDPsi_y1=m_SurfDPsi_FD_y1; pDPsi_y2=m_SurfDPsi_FD_y2;
	 pDPsi_z1=m_SurfDPsi_FD_z1; pDPsi_z2=m_SurfDPsi_FD_z2;
	 break;
      default:
	 return;
   }
   switch (l_sp) {
      case 0:
	 sstr << "_u.dat";
	 break;
      case 2:
	 sstr << "_d.dat";
	 break;
      default:
	 return;
   }

   MPI_File fh;
   if (MPI_File_open(m_cartcomm, sstr.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)!=MPI_SUCCESS)
      throw runtime_error("Cannot open output file " + sstr.str());

   if (!m_Rank) {
      int precision=sizeof(QPrecision);
      MPI_File_write(fh, &precision, 1, MPIInt, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nXGlb, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nYGlb, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nZGlb, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_X1Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_X2Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_Y1Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_Y2Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_Z1Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_Z2Vrtl, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_nABC, 1, MPILong, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dx, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dy, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_dz, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_OrigX, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_OrigY, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_OrigZ, 1, MPIQPrecision, MPI_STATUS_IGNORE);
      MPI_File_write(fh, &m_E0, 1, MPIQPrecision, MPI_STATUS_IGNORE);
   }

   MPI_Offset disp;
   MPI_Datatype global_array, local_array;
   int gsizes[3], lsizes[3], subsizes[3], starts[3];

   disp=sizeof(int)+sizeof(m_nXGlb)+sizeof(m_nYGlb)+sizeof(m_nZGlb)
      +sizeof(m_X1Vrtl)+sizeof(m_X2Vrtl)+sizeof(m_Y1Vrtl)+sizeof(m_Y2Vrtl)
      +sizeof(m_Z1Vrtl)+sizeof(m_Z2Vrtl)+sizeof(m_nABC)+sizeof(m_dx)
      +sizeof(m_dy)+sizeof(m_dz)+sizeof(m_OrigX)+sizeof(m_OrigY)
      +sizeof(m_OrigZ)+sizeof(m_E0);

   /************************ back+front surface data ************************/
   long incre=m_nYGlb*m_nZGlb*2*sizeof(QPrecision);

   lsizes[0]=m_Surf_x_ny; lsizes[1]=m_Surf_x_nz/4; lsizes[2]=4;
   subsizes[0]=m_nyCore; subsizes[1]=m_nzCore; subsizes[2]=2;
   starts[0]=0; starts[1]=0; starts[2]=l_sp;
   MPI_Type_create_subarray(3, lsizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &local_array);
   MPI_Type_commit(&local_array);
   gsizes[0]=m_nYGlb; gsizes[1]=m_nZGlb; gsizes[2]=2;
   starts[0]=m_y1Core; starts[1]=m_z1Core; starts[2]=0;
   MPI_Type_create_subarray(3, gsizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &global_array);
   MPI_Type_commit(&global_array);

   // back surface data
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, m_SurfPsi_x1, (m_flgx1Vrtl)?1:0, local_array, MPI_STATUS_IGNORE);
   disp += incre;
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, pDPsi_x1, (m_flgx1Vrtl)?1:0, local_array, MPI_STATUS_IGNORE);
   disp += incre;

   // front surface data
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, m_SurfPsi_x2, (m_flgx2Vrtl)?1:0, local_array, MPI_STATUS_IGNORE);
   disp += incre;
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, pDPsi_x2, (m_flgx2Vrtl)?1:0, local_array, MPI_STATUS_IGNORE);
   disp += incre;
   MPI_Type_free(&local_array);
   MPI_Type_free(&global_array);

   /************************ left+right surface data ************************/
   incre = m_nXGlb*m_nZGlb*2*sizeof(QPrecision);
   lsizes[0]=m_Surf_y_nx; lsizes[1]=m_Surf_y_nz/4; lsizes[2]=4;
   subsizes[0]=m_nxCore; subsizes[1]=m_nzCore; subsizes[2]=2;
   starts[0]=0; starts[1]=0; starts[2]=l_sp;
   MPI_Type_create_subarray(3, lsizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &local_array);
   MPI_Type_commit(&local_array);
   gsizes[0]=m_nXGlb; gsizes[1]=m_nZGlb; gsizes[2]=2;
   starts[0]=m_x1Core; starts[1]=m_z1Core; starts[2]=0;
   MPI_Type_create_subarray(3, gsizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &global_array);
   MPI_Type_commit(&global_array);

   // left surface data
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, m_SurfPsi_y1, (m_flgy1Vrtl)?1:0, local_array, MPI_STATUS_IGNORE);
   disp += incre;
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, pDPsi_y1, (m_flgy1Vrtl)?1:0, local_array, MPI_STATUS_IGNORE);
   disp += incre;

   // right surface data
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, m_SurfPsi_y2, (m_flgy2Vrtl)?1:0, local_array, MPI_STATUS_IGNORE);
   disp += incre;
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, pDPsi_y2, (m_flgy2Vrtl)?1:0, local_array, MPI_STATUS_IGNORE);
   disp += incre;
   MPI_Type_free(&local_array);
   MPI_Type_free(&global_array);

   /************************ bottom+top surface data ************************/
   incre = m_nXGlb*m_nYGlb*2*sizeof(QPrecision);
   lsizes[0]=m_Surf_z_nx; lsizes[1]=m_Surf_z_ny/4; lsizes[2]=4;
   subsizes[0]=m_nxCore; subsizes[1]=m_nyCore; subsizes[2]=2;
   starts[0]=0; starts[1]=0; starts[2]=l_sp;
   MPI_Type_create_subarray(3, lsizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &local_array);
   MPI_Type_commit(&local_array);
   gsizes[0]=m_nXGlb; gsizes[1]=m_nYGlb; gsizes[2]=2;
   starts[0]=m_x1Core; starts[1]=m_y1Core; starts[2]=0;
   MPI_Type_create_subarray(3, gsizes, subsizes, starts, MPI_ORDER_C, MPIQPrecision, &global_array);
   MPI_Type_commit(&global_array);

   // bottom surface data
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, m_SurfPsi_z1, (m_flgz1Vrtl)?1:0, local_array, MPI_STATUS_IGNORE);
   disp += incre;
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, pDPsi_z1, (m_flgz1Vrtl)?1:0, local_array, MPI_STATUS_IGNORE);
   disp += incre;

   // top surface data
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, m_SurfPsi_z2, (m_flgz2Vrtl)?1:0, local_array, MPI_STATUS_IGNORE);
   disp += incre;
   MPI_File_set_view(fh, disp, MPIQPrecision, global_array, "native", MPI_INFO_NULL);
   MPI_File_write_all(fh, pDPsi_z2, (m_flgz2Vrtl)?1:0, local_array, MPI_STATUS_IGNORE);
   MPI_Type_free(&local_array);
   MPI_Type_free(&global_array);

   MPI_File_close(&fh);
}

void CPSTD_MagScat::HandleError(int val, string id_str)
{
   stringstream sst;
   sst << "Error " << val << ": Process (" << m_xRank << \
      "," << m_yRank << "," << m_zRank << "), " \
      << id_str;
   throw runtime_error(sst.str());
}

void CPSTD_MagScat::WriteInitStatusToSTDOUT()
{
   stringstream sstr;
   sstr << "Rank(" << m_xRank << "," << m_yRank << "," << m_zRank << "):" << endl;
   sstr << "\tnProcs (" << m_nXProcs << "," << m_nYProcs << "," << \
      m_nZProcs << ")\tnThreads: " << m_nThread << endl;
   sstr << "\tnGlb(" << m_nXGlb << "," << m_nYGlb << "," << m_nZGlb << ")" << endl;
   sstr << "\tnABC, nSca, nKsi, nHalo (" << m_nABC << "," << m_nSca << \
      "," << m_nKsi << "," << m_nHalo << ")" << endl;
   sstr << "\tXABC(" << m_X1ABC << "," << m_X2ABC << "), YABC(" << \
      m_Y1ABC << "," << m_Y2ABC << "), ZABC(" << m_Z1ABC << "," << \
      m_Z2ABC << ")" << endl;
   sstr << "\tX1Ksi(" << m_X1aKsi << "," << m_X1bKsi << "), X2Ksi(" << \
      m_X2aKsi << "," << m_X2bKsi << ")" << endl;
   sstr << "\tY1Ksi(" << m_Y1aKsi << "," << m_Y1bKsi << "), Y2Ksi(" << \
      m_Y2aKsi << "," << m_Y2bKsi << ")" << endl;
   sstr << "\tZ1Ksi(" << m_Z1aKsi << "," << m_Z1bKsi << "), Z2Ksi(" << \
      m_Z2aKsi << "," << m_Z2bKsi << ")" << endl;
   sstr << "\tXVrtl(" << m_X1Vrtl << "," << m_X2Vrtl << "), YVrtl(" << \
      m_Y1Vrtl << "," << m_Y2Vrtl << "), ZVrtl(" << m_Z1Vrtl << \
      "," << m_Z2Vrtl << ")" << endl;
   sstr << "\tdtau,dx,dy,dz (" << m_dtau << "," << m_dx << "," << \
      m_dy << "," << m_dz << ")" << endl;
   sstr << "\txFFT(" << m_x1FFT << "," << m_x2FFT << "), yFFT(" << \
      m_y1FFT << "," << m_y2FFT << "), zFFT(" << m_z1FFT << \
      "," << m_z2FFT << ")" << endl;
   sstr << "\txCore(" << m_x1Core << "," << m_x2Core << "), yCore(" << \
      m_y1Core << "," << m_y2Core << "), zCore(" << m_z1Core << \
      "," << m_z2Core << ")" << endl;
   sstr << "\tpsi dimension (" << m_nx << "," << m_ny << "," << m_nz << \
      "), " << m_nyz << endl;
   sstr << "\txV(" << m_x1V << "," << m_x2V << "), yV(" << m_y1V << \
      "," << m_y2V << "), zV(" << m_z1V << "," << m_z2V << ")" << endl;
   sstr << "\tV dimension (" << m_nxV << "," << m_nyV << "," << \
      m_nzV << "), " << m_nyzV << endl;
   sstr << "\tOrigin (" << m_OrigX << ","<< m_OrigY << "," << \
      m_OrigZ << ")" << endl;
   sstr << "\ttheta,phi,E0,lambdabar0 (" << m_theta << "," << m_phi << \
      "," << m_E0 << "," << m_lambdabar0 << ")" << endl;
   sstr << "\tIncident spin: " << m_sp0[0] << ", " << m_sp0[1] << ", " << \
      m_sp0[2] << ", " << m_sp0[3] << endl;
   if (m_flgx1Ksi) {
      sstr << "\tflgx1Ksi (" << m_slab_x1.xa << "," << m_slab_x1.xb << \
	 "," << m_slab_x1.ya << "," << m_slab_x1.yb << "," << m_slab_x1.za << \
	 "," << m_slab_x1.zb << "), array dimension (" << m_slab_x1.nx << \
	 "," << m_slab_x1.ny << "," << m_slab_x1.nz << ")" << endl;
   }
   if (m_flgx2Ksi) {
      sstr << "\tflgx2Ksi (" << m_slab_x2.xa << "," << m_slab_x2.xb << \
	 "," << m_slab_x2.ya << "," << m_slab_x2.yb << "," << m_slab_x2.za << \
	 "," << m_slab_x2.zb << "), array dimension (" << m_slab_x2.nx << \
	 "," << m_slab_x2.ny << "," << m_slab_x2.nz << ")" << endl;
   }
   if (m_flgy1Ksi) {
      sstr << "\tflgy1Ksi (" << m_slab_y1.xa << "," << m_slab_y1.xb << \
	 "," << m_slab_y1.ya << "," << m_slab_y1.yb << "," << m_slab_y1.za << \
	 "," << m_slab_y1.zb << "), array dimension (" << m_slab_y1.nx << \
	 "," << m_slab_y1.ny << "," << m_slab_y1.nz << ")" << endl;
   }
   if (m_flgy2Ksi) {
      sstr << "\tflgy2Ksi (" << m_slab_y2.xa << "," << m_slab_y2.xb << \
	 "," << m_slab_y2.ya << "," << m_slab_y2.yb << "," << m_slab_y2.za << \
	 "," << m_slab_y2.zb << "), array dimension (" << m_slab_y2.nx << \
	 "," << m_slab_y2.ny << "," << m_slab_y2.nz << ")" << endl;
   }
   if (m_flgz1Ksi) {
      sstr << "\tflgz1Ksi (" << m_slab_z1.xa << "," << m_slab_z1.xb << \
	 "," << m_slab_z1.ya << "," << m_slab_z1.yb << "," << m_slab_z1.za << \
	 "," << m_slab_z1.zb << "), array dimension (" << m_slab_z1.nx << \
	 "," << m_slab_z1.ny << "," << m_slab_z1.nz << ")" << endl;
   }
   if (m_flgz2Ksi) {
      sstr << "\tflgz2Ksi (" << m_slab_z2.xa << "," << m_slab_z2.xb << \
	 "," << m_slab_z2.ya << "," << m_slab_z2.yb << "," << m_slab_z2.za << \
	 "," << m_slab_z2.zb << "), array dimension (" << m_slab_z2.nx << \
	 "," << m_slab_z2.ny << "," << m_slab_z2.nz << ")" << endl;
   }

#ifdef __IBH__
   sstr << "Using Integrated BH.  Weight:" << endl;
#else
   sstr << "Using SMOOTH4.  Weight:" << endl;
#endif
   for (long i=0; i<NWEIGHT; ++i)
      sstr << "w[" << i << "]=" << m_weight[i] << endl;

   cout << sstr.str() << endl;
}

CPSTD_MagScat::~CPSTD_MagScat()
{
   int i;

   Free_Aligned_Matrix_3D<QPrecision>(m_Psi0, m_nx, m_ny, m_nz);
   Free_Aligned_Matrix_3D<QPrecision>(m_Psi, m_nx, m_ny, m_nz);

   Free_Aligned_Vector<QPrecision>(m_g1, m_nxCore);
   Free_Aligned_Vector<QPrecision>(m_g2, m_nyCore);
   Free_Aligned_Vector<QPrecision>(m_g3, m_nzCore);

   if (m_flgx1Ksi)
      Free_Aligned_Matrix_3D<QPrecision>(m_Inc_x1, m_slab_x1.nx, m_slab_x1.ny, m_slab_x1.nz);
   if (m_flgx2Ksi)
      Free_Aligned_Matrix_3D<QPrecision>(m_Inc_x2, m_slab_x2.nx, m_slab_x2.ny, m_slab_x2.nz);
   if (m_flgy1Ksi)
      Free_Aligned_Matrix_3D<QPrecision>(m_Inc_y1, m_slab_y1.nx, m_slab_y1.ny, m_slab_y1.nz);
   if (m_flgy2Ksi)
      Free_Aligned_Matrix_3D<QPrecision>(m_Inc_y2, m_slab_y2.nx, m_slab_y2.ny, m_slab_y2.nz);
   if (m_flgz1Ksi)
      Free_Aligned_Matrix_3D<QPrecision>(m_Inc_z1, m_slab_z1.nx, m_slab_z1.ny, m_slab_z1.nz);
   if (m_flgz2Ksi)
      Free_Aligned_Matrix_3D<QPrecision>(m_Inc_z2, m_slab_z2.nx, m_slab_z2.ny, m_slab_z2.nz);

   if (m_nXProcs>1 && m_xTypeExch) MPI_Type_free(&m_xTypeExch);
   if (m_nYProcs>1 && m_yTypeExch) MPI_Type_free(&m_yTypeExch);
   if (m_nZProcs>1 && m_zTypeExch) MPI_Type_free(&m_zTypeExch);

   if (m_descx) {
      for (i=0; i<m_nThread; ++i)
	 if (m_descx[i]) DftiFreeDescriptor(&m_descx[i]);
      free(m_descx);
   }
   if (m_descy) {
      for (i=0; i<m_nThread; ++i)
	 if (m_descy[i]) DftiFreeDescriptor(&m_descy[i]);
      free(m_descy);
   }
   if (m_descz) {
      for (i=0; i<m_nThread; ++i)
	 if (m_descz[i]) DftiFreeDescriptor(&m_descz[i]);
      free(m_descz);
   }

   Free_MKL_Matrix_2D<MKLComplex>(m_vFFT, 0, m_nThread-1, 0,
	 max(max(m_nXFFT, m_nYFFT), m_nZFFT)-1);

   Free_Aligned_Vector<QPrecision>(m_k1x, m_nXFFT*2);
   Free_Aligned_Vector<QPrecision>(m_k1y, m_nYFFT*2);
   Free_Aligned_Vector<QPrecision>(m_k1z, m_nZFFT*2);
   Free_Aligned_Vector<QPrecision>(m_k2x, m_nXFFT);
   Free_Aligned_Vector<QPrecision>(m_k2y, m_nYFFT);
   Free_Aligned_Vector<QPrecision>(m_k2z, m_nZFFT);

   if (m_flgx1Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfPsi_x1, m_Surf_x_ny, m_Surf_x_nz);
   if (m_flgx2Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfPsi_x2, m_Surf_x_ny, m_Surf_x_nz);
   if (m_flgy1Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfPsi_y1, m_Surf_y_nx, m_Surf_y_nz);
   if (m_flgy2Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfPsi_y2, m_Surf_y_nx, m_Surf_y_nz);
   if (m_flgz1Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfPsi_z1, m_Surf_z_nx, m_Surf_z_ny);
   if (m_flgz2Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfPsi_z2, m_Surf_z_nx, m_Surf_z_ny);
   if (m_flgx1Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_PS_x1, m_Surf_x_ny, m_Surf_x_nz);
   if (m_flgx2Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_PS_x2, m_Surf_x_ny, m_Surf_x_nz);
   if (m_flgy1Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_PS_y1, m_Surf_y_nx, m_Surf_y_nz);
   if (m_flgy2Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_PS_y2, m_Surf_y_nx, m_Surf_y_nz);
   if (m_flgz1Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_PS_z1, m_Surf_z_nx, m_Surf_z_ny);
   if (m_flgz2Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_PS_z2, m_Surf_z_nx, m_Surf_z_ny);
   if (m_flgx1Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_FD_x1, m_Surf_x_ny, m_Surf_x_nz);
   if (m_flgx2Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_FD_x2, m_Surf_x_ny, m_Surf_x_nz);
   if (m_flgy1Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_FD_y1, m_Surf_y_nx, m_Surf_y_nz);
   if (m_flgy2Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_FD_y2, m_Surf_y_nx, m_Surf_y_nz);
   if (m_flgz1Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_FD_z1, m_Surf_z_nx, m_Surf_z_ny);
   if (m_flgz2Vrtl) Free_Aligned_Matrix_2D<QPrecision>(m_SurfDPsi_FD_z2, m_Surf_z_nx, m_Surf_z_ny);

   if (m_flgV)
      Free_Aligned_Matrix_3D<QPrecision>(m_VB_i, m_nxV, m_nyV, m_nzV);
}
