CC ?= clang
CFLAGS ?= -fsanitize=undefined
MKLROOT ?= /opt/intel/mkl


all: loadfits fits_to_evecs lapacke_dsyevr_row lapacke_dsyevr_col test
loadfits: loadfits.c
	$(CC) $(CFLAGS) -lcfitsio -lm -o loadfits loadfits.c

lapacke_dsyevr_row: lapacke_dsyevr_row.c
	$(CC) $(CFLAGS) \
		-DMKL_ILP64 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
		-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl \
		-o lapacke_dsyevr_row lapacke_dsyevr_row.c

lapacke_dsyevr_col: lapacke_dsyevr_col.c
	$(CC) $(CFLAGS) \
		-DMKL_ILP64 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
		-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl \
		-o lapacke_dsyevr_col lapacke_dsyevr_col.c

lapacke_dsyevr_work_col: lapacke_dsyevr_work_col.c
	$(CC) $(CFLAGS) \
		-DMKL_ILP64 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
		-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl \
		-o lapacke_dsyevr_work_col lapacke_dsyevr_work_col.c

fits_to_evecs: fits_to_evecs.c
	$(CC) $(CFLAGS) \
		-DMKL_ILP64 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
		-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl \
		-lcfitsio \
		-o fits_to_evecs fits_to_evecs.c

test_runner: doodads.h tests/test_runner.c
	cd ./tests && \
	$(CC) $(CFLAGS) \
		-DMKL_ILP64 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
		-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lcfitsio \
		-o test_runner test_runner.c

test: test_runner
	cd ./tests && \
	./test_runner