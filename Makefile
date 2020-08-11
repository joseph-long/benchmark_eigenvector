CC = gcc
CFLAGS ?= -g3 -fsanitize=undefined -Wall -I$(HOME)/.local/include -L$(HOME)/.local/lib
MKLROOT ?= /opt/intel/mkl
TEMPDIR := $(shell mktemp -d)

all: fits_to_evecs test

fits_to_evecs: fits_to_evecs.c doodads.h
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

python_cross_check: fits_to_evecs
	./fits_to_evecs \
		--method mkl_syevr \
		--warmups 2 \
		--iterations 1 \
		--ramp 1 \
		--outprefix $(TEMPDIR)/test \
		tests/data/noise_20_30.fits
	python \
		tests/compare_evecs_evals.py \
		tests/data/noise_20_30.fits \
		$(TEMPDIR)/test_cov.fits \
		$(TEMPDIR)/test_evecs.fits \
		$(TEMPDIR)/test_evals.fits

test: test_runner python_cross_check
	cd ./tests && \
	./test_runner

clean:
	rm -f fits_to_evecs
	rm -f tests/test_runner

.PHONY: all test python_cross_check clean
