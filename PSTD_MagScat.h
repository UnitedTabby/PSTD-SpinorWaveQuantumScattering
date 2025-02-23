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

#ifndef __CPSTD_MagScat__
#define __CPSTD_MagScat__

#include <string>
#include "mkl.h"
#include "mkl_types.h"
#include "mkl_dfti.h"
#include <mpi.h>
#include "run_environment.h"

typedef struct {
   long xa;
   long xb;
   long ya;
   long yb;
   long za;
   long zb;
   long nx;
   long ny;
   long nz;
} slab;

// calculating the surface normal derivative of the wave function on
// the virtual surface: PS, using the pseudospectral method;
// FD: using the finite difference method
enum DerivativeType {PS, FD};

// 8 doubles of weight factors, fill exactly 64-byte cache line
#define NWEIGHT		8

class CPSTD_MagScat
{
   private:
      QPrecision *m_Psi0, *m_Psi;
      // g=U0/Cosh^2(alpha*n)
      QPrecision *m_g1, *m_g2, *m_g3;
      alignas(CACHE_LINE) QPrecision m_weight[NWEIGHT];

      // overlapping domain decomposition:
      // To balance the computation between different nodes, all domains
      // should have the same FFT length.  Thus, the domain
      // decomposition is generated from the pre-determined FFT length.  Why?
      // Because FFT is the central computation of PSTD, and 
      // the Intel MKL FFT runs faster when the length
      // permits factorization into powers of 2, 3, 5, 7, or 11, the so-called
      // radices.  It is not easy to set the domain and halo sizes to make
      // the FFT length fit to the radices rule; it is much easier to preset
      // the optimal FFT length and halo size, and deduce the domain size.
      // 
      // There is a utility length-advisor.cpp, provided by Intel, to determine
      // the optimal FFT length for the targeted model size.  The advised length
      // may not be the exact size, but is close.

      // Global Entities: all nodes (processes) share the same values,
      // variables starting with an upper case character are global ones
      // defined in the entire lattice; 
      // an X/Y/Z variable contains the global indice of the
      // entire lattice
      int m_nXProcs, m_nYProcs, m_nZProcs;      // global number of processes
      int m_nThread;                            // number of OpenMP threads
      long m_nXGlb, m_nYGlb, m_nZGlb;            // sizes of the entire lattice
      long m_nSca;	// thickness of scattered field zone
      long m_nKsi;      // thickness of the transition layer, i.e., Ksi region
      long m_nHalo;	// thickness of overlapping region, i.e. the halo
      long m_nXFFT, m_nYFFT, m_nZFFT;	// number of FFT grids of each domain, should be even values
					// for the balance of computation, all nodes share the same values
      long m_nABC;	// thickness of the numerical absorbing boundary
      QPrecision m_CAP_U0, m_CAP_alpha;	// parameters for the absorbing boundary condition (ABC)
      long m_X1ABC, m_X2ABC, m_Y1ABC, m_Y2ABC, m_Z1ABC, m_Z2ABC;	// the global indices (inclusive) of the start and end of ABC
      long m_XInj, m_YInj, m_ZInj;			// Injection corner of the incident wave
      long m_X1aKsi, m_X1bKsi, m_X2aKsi, m_X2bKsi;	// X indices (inclusive) of the transition layer, i.e., Ksi region
      long m_Y1aKsi, m_Y1bKsi, m_Y2aKsi, m_Y2bKsi;	// Y indices (inclusive) of the transition layer, i.e., Ksi region
      long m_Z1aKsi, m_Z1bKsi, m_Z2aKsi, m_Z2bKsi;	// Z indices (inclusive) of the transition layer, i.e., Ksi region
      long m_X1Vrtl, m_X2Vrtl, m_Y1Vrtl, m_Y2Vrtl, m_Z1Vrtl, m_Z2Vrtl; //
      			// Virtual surface indices, chosen to be at the middle of the scattered-field zone.
      			// So better to set m_nSca an odd value 
      QPrecision m_dtau, m_dx, m_dy, m_dz;       // unit in radians, for example, 5 grids per wavelength means m_dx=2pi/5, etc.
      QPrecision m_OrigX, m_OrigY, m_OrigZ;	// grid location of origin O, not long type in case the total number of 
      						// grids is even. we set it to the center of the entire lattice.
      QPrecision m_alpha_2dtau_i;		// correction-factor-for-numerical-dispersion*I*2*dtau
      QPrecision m_theta, m_phi, m_E0, m_lambdabar0, m_sp0[4];	// direction (theta, phi), energy and wavelength of the incident wave
      							// E in micro-eV, lambdabar=lambda/(2pi) in unit of nm/radian


      // Local Entities: different nodes (processes) contain different values
      // variables starting with a lower case character are local entities,
      // an x/y/z variable: local variable containing global index,
      // an i/j/k variable: local variable containing local index (i.e., index in the domain),
      long m_nx, m_ny, m_nz, m_nyz;	// dimensions of local zone; due to padding, m_nz may be larger than FFT zone
					// padding is added to align the 0-indiced element of the fastest varying dimension
                                        // to the memory CACHE_LINE boundary, for better performance
      long m_x1Core, m_x2Core, m_y1Core, m_y2Core, m_z1Core, m_z2Core;	// global indices of non-overlapping zone
      long m_i1Core, m_i2Core, m_j1Core, m_j2Core, m_k1Core, m_k2Core;	// local indices of non-overlapping zone
									// the x- and i- indices refer to the same grid
                                                                        // same for (y- and j-), (z- and k-)
      long m_nxCore, m_nyCore, m_nzCore;	// number of Core grids
      long m_x1FFT, m_x2FFT, m_y1FFT, m_y2FFT, m_z1FFT, m_z2FFT; // global indices of FFT zone (overlapping + non-overlapping)

      // Ksi related, local entities
      bool m_flgx1Ksi, m_flgx2Ksi;	// flags: this domain's core contains Ksi grids (1) or not (0)
      bool m_flgy1Ksi, m_flgy2Ksi;
      bool m_flgz1Ksi, m_flgz2Ksi;
      slab m_slab_x1, m_slab_x2;	// end points of crossing Ksi zone
      slab m_slab_y1, m_slab_y2;	// 6 slabs: back(x1), front(x2), left(y1),
      slab m_slab_z1, m_slab_z2;	//	    right(y2), bottom(z1), top(z2)
      QPrecision *m_Inc_x1, *m_Inc_x2;	// coefficients for the incident wave contributions, where
      QPrecision *m_Inc_y1, *m_Inc_y2;	// *m_Inc_x1 corresponds to the back slab, etc.
      QPrecision *m_Inc_z1, *m_Inc_z2;

      // MPI related
      MPI_Comm m_cartcomm;                      // 3D topology of the processor-to-domain mapping
      MPI_Status m_status;
      int m_Rank, m_xRank, m_yRank, m_zRank;    // rank indices of this process
      int m_xaRank, m_xbRank, m_yaRank, m_ybRank, m_zaRank, m_zbRank;   // rank indices of neighboring processes
      MPI_Datatype m_xTypeExch, m_yTypeExch, m_zTypeExch;       // overlapping data structures for communication with neighbors
      long m_xaSendOffset, m_xaRecvOffset, m_xbSendOffset, m_xbRecvOffset;	// offsets for the starting indices of the exchange zones
      long m_yaSendOffset, m_yaRecvOffset, m_ybSendOffset, m_ybRecvOffset;
      long m_zaSendOffset, m_zaRecvOffset, m_zbSendOffset, m_zbRecvOffset;

      // mkl FFT related
      DFTI_DESCRIPTOR_HANDLE *m_descx, *m_descy, *m_descz;
      QPrecision *m_k1x, *m_k1y, *m_k1z;	// for FFT operation of 1st order derivatives
      QPrecision *m_k2x, *m_k2y, *m_k2z;	// for FFT operation of 2nd order derivatives
      MKLComplex **m_vFFT;

      // virtual surface related
      long m_nTau;	// the time index of the latest wave function 
      long m_nSur;			// number of times that surface terms are accumulated
      bool m_flgx1Vrtl, m_flgx2Vrtl;	// flags: this domain's core contains virtual surface (1) or not (0)
      bool m_flgy1Vrtl, m_flgy2Vrtl;
      bool m_flgz1Vrtl, m_flgz2Vrtl;
      // the wave function values on the six surfaces.  memory allocation in rank 0 is different from the other ranks,  the results are gathered to and saved in rank 0.
      QPrecision *m_SurfPsi_x1, *m_SurfPsi_x2;
      QPrecision *m_SurfPsi_y1, *m_SurfPsi_y2;
      QPrecision *m_SurfPsi_z1, *m_SurfPsi_z2;
      // the derivatives along the external normal of the six surfaces, calculated using PS approach
      QPrecision *m_SurfDPsi_PS_x1, *m_SurfDPsi_PS_x2;
      QPrecision *m_SurfDPsi_PS_y1, *m_SurfDPsi_PS_y2;
      QPrecision *m_SurfDPsi_PS_z1, *m_SurfDPsi_PS_z2;
      // the derivatives along the external normal of the six surfaces, calculated using FD approach
      QPrecision *m_SurfDPsi_FD_x1, *m_SurfDPsi_FD_x2;
      QPrecision *m_SurfDPsi_FD_y1, *m_SurfDPsi_FD_y2;
      QPrecision *m_SurfDPsi_FD_z1, *m_SurfDPsi_FD_z2;
      long m_Surf_x_ny, m_Surf_x_nz;	// the array sizes of the surface terms
      long m_Surf_y_nx, m_Surf_y_nz;
      long m_Surf_z_nx, m_Surf_z_ny;

      // quantum potential related
      QPrecision *m_VB_i;	// Vector potential at each grid
      bool m_flgV;	// flags: this domain's core contains total field grids (1) or not (0)
      long m_x1V, m_x2V, m_y1V, m_y2V, m_z1V, m_z2V;	// end indices of the total field grids in this domain
      long m_nxV, m_nyV, m_nzV, m_nyzV;	// sizes of m_VB_i in this domain

      // the naming of the member functions are self-explaintory
      void SetupCartComm(int nx_proc, int ny_proc, int nz_proc, MPI_Comm comm);
      void DomainDecomposition();
      void InitMPIExchange();
      void InitFFT();
      void InitABC();
      void InitTS();
      void InitPsi();
      void InitV(void (*V)(QPrecision *, QPrecision, QPrecision, QPrecision, QPrecision));
      void InitGradKsi();
      void InitVirtualSurfaces();
      void UpdateX(int myid, long j, long k);
      void UpdateY(int myid, long i, long k);
      void UpdateZ(int myid, long i, long j);
      void ExchangeData();
      void SavePsi_XPlane(std::string file_name_prefix, long l_sp, long i0);
      void SavePsi_YPlane(std::string file_name_prefix, long l_sp, long j0);
      void SavePsi_ZPlane(std::string file_name_prefix, long l_sp, long k0);
      void FinalizeSurfaceTerms();
      void SaveSurfaceTerms(std::string file_name_prefix, long l_sp, DerivativeType PSvsFD);
      void ResetSurfaceTerms();
      void HandleError(int val, std::string id_str);
      void WriteInitStatusToSTDOUT();
      
   public:
      CPSTD_MagScat();
      ~CPSTD_MagScat();

      void Init(int nx_proc, int ny_proc, int nz_proc,
	    QPrecision dtau, QPrecision dx, QPrecision dy, QPrecision dz,
	    long nx_fft, long ny_fft, long nz_fft, long n_abc, long n_sca, long n_ksi,
	    long n_halo, QPrecision theta, QPrecision phi, QPrecision E0, QPrecision *sp0,
	    QPrecision CAP_U0, QPrecision CAP_alpha,
	    void (*func) (QPrecision *, QPrecision, QPrecision, QPrecision, QPrecision),
	    MPI_Comm comm);
      void MPILoadSnapshot_all(std::string snapshot, 
	    void (*func) (QPrecision *, QPrecision, QPrecision, QPrecision, QPrecision),
	    MPI_Comm comm);
      void MPISaveSnapshot_all(std::string snapshot_prefix);
      void MPILoadSnapshot(std::string snapshot, 
	    void (*func) (QPrecision *, QPrecision, QPrecision, QPrecision, QPrecision),
	    MPI_Comm comm);
      void MPISaveSnapshot(std::string snapshot_prefix);
      void Update();
      long IterationsToAccumulateSurfaceTerms();
      void AccumulateSurfaceTerms();
      void SaveVirtualSurfaces(std::string file_name_prefix);
      void Save_g();
      void SavePsi_XPlane(std::string file_name_prefix, QPrecision x0);
      void SavePsi_XPlane(std::string file_name_prefix, long i0);
      void SavePsi_YPlane(std::string file_name_prefix, QPrecision y0);
      void SavePsi_YPlane(std::string file_name_prefix, long j0);
      void SavePsi_ZPlane(std::string file_name_prefix, QPrecision z0);
      void SavePsi_ZPlane(std::string file_name_prefix, long k0);
};

#endif
