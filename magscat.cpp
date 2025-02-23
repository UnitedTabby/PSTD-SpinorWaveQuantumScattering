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
#include <string>
#include <stdexcept>
#include <string.h>
#include <mpi.h>
#include <sys/time.h>
#include "run_environment.h"
#include "PSTD_MagScat.h"
#include "utility.h"

#define TWOPI	6.283185307179586476925286766559005768394

using namespace std;

#define FREAD(a, b, c)	{if (!((a).good() && ((a).read((char *) (b), (c))).good())) \
      {(a).close(); throw runtime_error("Error: incomplete snapshot data file");}}

void show_usage(char *app);
void load_input(string& openname, int &flag_ldsnp, string& ldsnp_name,
      int &nx_proc, int &ny_proc, int &nz_proc, QPrecision &dtau,
      QPrecision &dx, QPrecision &dy, QPrecision &dz, long &nx_fft, long &ny_fft, long &nz_fft,
      long &n_abc, long &n_sca, long &n_ksi, long &n_halo, QPrecision &theta, QPrecision &phi,
      QPrecision &E0, QPrecision *sp0, QPrecision &CAP_U0, QPrecision &CAP_alpha,
      long &Iter, int &flag_svsnp, string &svsnp_prefix, string& surfname_prefix);

int main(int argc, char *argv[])
{
   /* the number of threads per processor will be set via the num_threads environment
    * variable on the make command-line;
    */

   /* read in parameters from input file */
   /* get the numbers of processors nx_proc, ny_proc, nz_proc
    * in x, y, z directions;
    *
    * get dtau, dx, dy, dz in unit of radian;
    * 
    * get nx_fft, ny_fft, nz_fft;
    * (for the sake of FFT performance, these are carefully chosen beforehand)
    * (the total number of grids in the model, nx_glb, ny_glb, nz_glb are derived)
    * 
    * get the thicknesses of pml, scattered-field region, ksi, domain overlapping;
    * (the positions of ksi region, virtual surfaces are derived)
    *
    * get the direction (theta,phi) and energy of the incident wave;
    *
    * get the ABC boundary CAP_U0 and CAP_alpha;
    *
    * get the number of total iterations to run;
    *
    * get the file name to save the surface terms;
    *
    * Error code: 12, i.e., ENOMEM, memory allocation error, insufficient memory;
    * 		  22, i.e., EINVAL, memory allocation error, invalid argument;
    * 		  2, the 3D process topology mismatches the total number of processes
    * 		  3, file I/O error;
    * 		  4, incomplete file for model parameters
    */

   string openname;
   for (int i=1; i<argc; ++i) {
      string arg=argv[i];
      if ((arg=="-h") || (arg=="--help")) {
	 show_usage(argv[0]);
	 return 0;
      }
      if (arg=="-i") {
	 if (i<argc-1) 
	    openname=argv[i+1];
	 if (i==argc-1 || openname[0]=='-') {
	    cerr << "Error: -i requires an argument" << endl;
	    show_usage(argv[0]);
	    return 1;
	 }
      }
   }

   if (openname.empty()) {
      cerr << "Error: mandatory -i FILE for model paramters" << endl;
      show_usage(argv[0]);
      return 1;
   }

   MPI_Init(&argc, &argv);

   try {
      int nx_proc, ny_proc, nz_proc;
      QPrecision dtau, dx, dy, dz;
      long nx_fft, ny_fft, nz_fft, n_abc, n_sca, n_ksi, n_halo;
      QPrecision theta, phi, E0, sp0[4], CAP_U0, CAP_alpha;
      long Iter;  // total iterations to run
      int flag_ldsnp, flag_svsnp;
      string ldsnp_name, svsnp_prefix;	// file names to load/save snapshot
      string surfname_prefix;  // filename to save the surface terms
      CPSTD_MagScat pstd_magscat;

      int rank, cnt;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (!rank) {
	 load_input(openname, flag_ldsnp, ldsnp_name, nx_proc, ny_proc, nz_proc,
	       dtau, dx, dy, dz, nx_fft, ny_fft, nz_fft, n_abc, n_sca, n_ksi, n_halo,
	       theta, phi, E0, sp0, CAP_U0, CAP_alpha,
	       Iter, flag_svsnp, svsnp_prefix, surfname_prefix);
      }
      // MPI standard v3.0 section 5.1: no guarantee of synchronization
      // for collective communication routines.  apply barrier here
      MPI_Bcast(&flag_ldsnp, 1, MPIInt, 0, MPI_COMM_WORLD);
      MPI_Bcast(&Iter, 1, MPI_LONG, 0, MPI_COMM_WORLD);
      MPI_Bcast(&flag_svsnp, 1, MPIInt, 0, MPI_COMM_WORLD);
      if (flag_svsnp) {
	 if (!rank) cnt=svsnp_prefix.length();
	 MPI_Bcast(&cnt, 1, MPIInt, 0, MPI_COMM_WORLD);
	 char *str=new char[cnt+1];
	 str[cnt]='\0';
	 if (!rank) strcpy(str, svsnp_prefix.c_str());
	 MPI_Bcast(str, cnt, MPI_CHAR, 0, MPI_COMM_WORLD);
	 if (rank) svsnp_prefix=str;
	 delete [] str;
      }
      {
	 if (!rank) cnt=surfname_prefix.length();
	 MPI_Bcast(&cnt, 1, MPIInt, 0, MPI_COMM_WORLD);
	 char *str=new char[cnt+1];
	 str[cnt]='\0';
	 if (!rank) strcpy(str, surfname_prefix.c_str());
	 MPI_Bcast(str, cnt, MPI_CHAR, 0, MPI_COMM_WORLD);
	 if (rank) surfname_prefix=str;
	 delete [] str;
      }
      if (!flag_ldsnp) {
	 MPI_Bcast(&nx_proc, 1, MPIInt, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&ny_proc, 1, MPIInt, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&nz_proc, 1, MPIInt, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&dtau, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&dx, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&dy, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&dz, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&nx_fft, 1, MPILong, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&ny_fft, 1, MPILong, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&nz_fft, 1, MPILong, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&n_abc, 1, MPILong, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&n_sca, 1, MPILong, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&n_ksi, 1, MPILong, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&n_halo, 1, MPILong, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&theta, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&phi, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&E0, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
	 MPI_Bcast(sp0, 4, MPIQPrecision, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&CAP_U0, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
	 MPI_Bcast(&CAP_alpha, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
	 pstd_magscat.Init(nx_proc, ny_proc, nz_proc, dtau, dx, dy, dz,
	       nx_fft, ny_fft, nz_fft, n_abc, n_sca, n_ksi, n_halo,
	       theta, phi, E0, sp0, CAP_U0,
	       CAP_alpha, Potential, MPI_COMM_WORLD);
      } else {
	 int cnt;
	 if (!rank) cnt=ldsnp_name.length();
	 MPI_Bcast(&cnt, 1, MPIInt, 0, MPI_COMM_WORLD);
	 char *str=new char[cnt+1];
	 str[cnt]='\0';
	 if (!rank) strcpy(str, ldsnp_name.c_str());
	 MPI_Bcast(str, cnt, MPI_CHAR, 0, MPI_COMM_WORLD);
	 if (rank) ldsnp_name=str;
	 delete [] str;
	 pstd_magscat.MPILoadSnapshot(ldsnp_name, Potential, MPI_COMM_WORLD);
      }

      MPI_Barrier(MPI_COMM_WORLD);

      pstd_magscat.Save_g();
      pstd_magscat.SavePsi_XPlane("Init", 0.0);
      pstd_magscat.SavePsi_YPlane("Init", 0.0);
      pstd_magscat.SavePsi_ZPlane("Init", 0.0);

      double tstart=MPI_Wtime();
      long iter_accum_duration=pstd_magscat.IterationsToAccumulateSurfaceTerms();
      long iter_th;
      bool flgAccum=false;
      for (long i=1; i<=Iter; ++i) {
	 if ( !(rank || Iter>127 && i%(Iter>>7)) )
	    cout << "Iter=" << i << ", clock time: " <<
	       MPI_Wtime()-tstart << endl;
	 
	 pstd_magscat.Update();
	 if (!(i%(Iter/10))) {
	    pstd_magscat.SavePsi_XPlane("Iter", 0.0);
	    pstd_magscat.SavePsi_YPlane("Iter", 0.0);
	    pstd_magscat.SavePsi_ZPlane("Iter", 0.0);
	 }

	 if (!flgAccum) {
	    if (i+iter_accum_duration-1>=Iter) {
	       iter_th=Iter;
	       flgAccum=true;
	    } else if (i+iter_accum_duration == 20001 || i+iter_accum_duration == 40001
		  || i+iter_accum_duration == 60001) {
	       iter_th=i+iter_accum_duration-1;
	       flgAccum=true;
	    }
	 }
	 if (flgAccum) {
	    pstd_magscat.AccumulateSurfaceTerms();
	    if (i==iter_th) {
	       pstd_magscat.SaveVirtualSurfaces(surfname_prefix);
	       flgAccum=false;
	    }
	 }
      }

      if (flag_svsnp) pstd_magscat.MPISaveSnapshot(svsnp_prefix);
   } catch (exception const &e) {
      cerr << e.what() << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
   }

   MPI_Finalize();
}

void show_usage(char *app)
{
   cout << "Usage: " << app << " -np # -i FILE\n" << \
      "where, #: number of processes; FILE: input file for model parameters" << endl;
}

void load_input(string &openname, int &flag_ldsnp, string &ldsnp_name,
      int &nx_proc, int &ny_proc, int &nz_proc, QPrecision &dtau, QPrecision &dx,
      QPrecision &dy, QPrecision &dz, long &nx_fft, long &ny_fft, long &nz_fft,
      long &n_abc, long &n_sca, long &n_ksi, long &n_halo, QPrecision &theta, QPrecision &phi,
      QPrecision &E0, QPrecision *sp0, QPrecision &CAP_U0, QPrecision &CAP_alpha,
      long &Iter, int &flag_svsnp, string &svsnp_prefix, string &surfname_prefix)
{
   std::ifstream ifs;
   string aline;
   stringstream sstr;

   int count=0;
   flag_svsnp=0;

   ifs.open(openname.c_str());
   if (!ifs.is_open())
      throw runtime_error("Error 3: Cannot open input file " + openname);

   if (getaline(ifs,aline)) {
      sstr.clear();
      sstr.str(aline);
      sstr >> flag_ldsnp;
      count++;
   }

   if (flag_ldsnp) {
      if (getaline(ifs,aline)) {
	 ldsnp_name=aline;
	 count++;
      }
      
      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> Iter;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 surfname_prefix=aline;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> flag_svsnp;
	 if (flag_svsnp) count++;
      }

      if (flag_svsnp && getaline(ifs,aline)) {
	 svsnp_prefix=aline;
	 count++;
      }

      ifs.close();
      if ((!flag_svsnp && count <4) || (flag_svsnp && count<6)) {
	 throw runtime_error("Error 4: incomplete parameter file "+openname);
      }
   } else {
      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> nx_proc >> ny_proc >> nz_proc;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> dtau >> dx >> dy >> dz;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> nx_fft >> ny_fft >> nz_fft;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> n_abc >> n_sca >> n_ksi >> n_halo;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> theta >> phi;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> E0;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> sp0[0] >> sp0[1] >> sp0[2] >> sp0[3];
	 QPrecision tmp=SQRT(sp0[0]*sp0[0]+sp0[1]*sp0[1]+sp0[2]*sp0[2]+sp0[3]*sp0[3]);
	 sp0[0] /= tmp;
	 sp0[1] /= tmp;
	 sp0[2] /= tmp;
	 sp0[3] /= tmp;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> CAP_U0;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> CAP_alpha;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> Iter;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 surfname_prefix=aline;
	 count++;
      }

      if (getaline(ifs,aline)) {
	 sstr.clear();
	 sstr.str(aline);
	 sstr >> flag_svsnp;
	 if (flag_svsnp) count++;
      }

      if (flag_svsnp && getaline(ifs,aline)) {
	 svsnp_prefix=aline;
	 count++;
      }

      if ((!flag_svsnp && count<12) || (flag_svsnp && count<14)) {
	 ifs.close();
	 throw runtime_error("Error 4: incomplete parameter file " + openname);
      }

      ifs.close();
   }
}
