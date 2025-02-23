# PSTD-SpinorWaveQuantumScattering
This software solves the magnetic neutron scattering in the Fresnel region and the far field.  The neutron's spin degree of freedom is represented by a spinor, and the Schrodinger equation of the spinor wavefunction is under investigation.  This is much more complicated than two independent scalar Schrodinger equations, due to the coupling between the two spin components.  The framework of absorbing boundary condition, total-field/scattered-field, and the pseudo-spectral operation is similar to that of the scalar wave theory, but with much more coding work.  It turns out the spinor calculation is much more sensitive to the accuracy of the node-to-node data exchange and the subsequent halo-data regularization than the scalar one is.  We have improved both afterwards.

The spinor wave solver takes much longer to run than the scalar one.  This software pacakge includes two sets of member functions, (1) MPILoadSnapshot_all and MPISaveSnapshot_all, (2) MPILoadSnapshot and MPISaveSnapshot.  Both pairs can take a snapshot of the memory space, and save the status to storage disks.  A new run can pick up what was left previously and continue the time iteration from there.  This serves in two situations: the preset number of iteration is not enough, or the supercomputer breaks down (The manager of our supercompting account specifically advises us to save middle results periodically, and the breaking-down rate is not near zero).  Function pair (1) contain no redundant data, but loads very slowly; whereas function pair (2) stores the useless halo data, but loads much faster.  It is a trade-off between CPU time and storage space.

Details about the theoretical derivation and the parallel algorithm for this software can be found in the paper arXiv:2412.14586[quant-ph], https://doi.org/10.48550/arXiv.2412.14586.

This package requires the Intel OneAPI with the MPI and MKL libraries.  The code is programmed with MPI-OpenMP-SIMD-vectorization hybrid parallelization. The output is the wave function and its surface normal derivative on the virtual surface, and the up and the down spin components are separately stored in different destination files.  Because they are decoupled outside the total-field zone, the two files can be safely processed individually by the near-to-distant-field transformation.  Two example input files are provided, one for a fresh run, and the other for a continuous run.  On a workstation of two CPUs and 10 cores/CPU, typical commands are (show_init_stauts is optional),

for compiling:

    num_threads=10 show_init_status=__INIT_STATUS__ make dmagscat 

for execution:

    mpirun -np 2 -iface lo ./dmagscat -i InputExample1.txt

The source code is copyrighted by: Kun Chen, Shanghai Institute of Optics and
Fine Mechanics, Chinese Academy of Sciences

and it is distributed under the terms of the MIT license.

Please see LICENSE file for details.
