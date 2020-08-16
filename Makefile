CC = gcc
MKLROOT ?= /opt/intel/mkl
CFLAGS ?=  -DMKL_ILP64 -m64 -I$(MKLROOT)/include -g3 -fsanitize=undefined -Wall -I$(HOME)/.local/include -L$(HOME)/.local/lib  -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
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
		--outprefix $(TEMPDIR)/test1 \
		tests/data/noise_20_30.fits
	python \
		tests/compare_evecs_evals.py \
		tests/data/noise_20_30.fits \
		$(TEMPDIR)/test1_cov.fits \
		$(TEMPDIR)/test1_evecs.fits \
		$(TEMPDIR)/test1_evals.fits \
		20
	./fits_to_evecs \
		--method mkl_syevr \
		--warmups 2 \
		--iterations 1 \
		--number_of_eigenvectors 5 \
		--outprefix $(TEMPDIR)/test2 \
		tests/data/noise_20_30.fits
	python \
		tests/compare_evecs_evals.py \
		tests/data/noise_20_30.fits \
		$(TEMPDIR)/test2_cov.fits \
		$(TEMPDIR)/test2_evecs.fits \
		$(TEMPDIR)/test2_evals.fits \
		5

unit_tests: test_runner
	cd ./tests && \
	./test_runner -v | ./greenest.awk

test: unit_tests python_cross_check

clean:
	rm -f fits_to_evecs
	rm -f tests/test_runner

.PHONY: all test python_cross_check clean
