MPIICPX=mpiicpx -std=c++17
SCATFLAGS=-xALDERLAKE -O3 -qopenmp -qopt-report -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -liomp5 -lpthread
MAGSCATTER=PSTD_MagScat.cpp potential.cpp utility.cpp magscat.cpp
HEADER_MAGSCATTER=PSTD_MagScat.h run_environment.h utility.h
num_threads?=1

fmagscat:	$(MAGSCATTER) $(HEADER_MAGSCATTER)
	echo =======================make fmagscat=======================
	$(MPIICPX) $(SCATFLAGS) -D__USE_FLOAT__ -D__OMP_NUM_THREADS__=$(num_threads) -D$(show_init_status) -D$(smooth_profile) -o fmagscat $(MAGSCATTER) -lm

dmagscat:	$(MAGSCATTER) $(HEADER_MAGSCATTER)
	echo =======================make dmagscat=======================
	$(MPIICPX) $(SCATFLAGS) -D__OMP_NUM_THREADS__=$(num_threads) -D$(show_init_status) -D$(smooth_profile) -o dmagscat $(MAGSCATTER) -lm

clean:
	rm -f fmagscat dmagscat *.optrpt *.opt.yaml
