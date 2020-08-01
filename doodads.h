#ifndef DOODADS_H_INCLUDED
#define DOODADS_H_INCLUDED
#include <time.h>
#include "fitsio.h"
#include "mkl.h"
#include <math.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>

#ifdef DD_DEBUG
void dd_debug(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}
#else
void dd_debug(const char *fmt, ...) {
    return;
}
#endif

// Column-major matrix
typedef struct dd_Matrix {
    long rows;
    long cols;
    double *data;
} dd_Matrix;

void dd_print_matrix(dd_Matrix *matrix)
{
    dd_debug("A is %li x %li\n", matrix->rows, matrix->cols);
    int i, j;
    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->cols; j++)
        {
            dd_debug("A[%li,%li] = %6.6e\t", i, j, matrix->data[j * matrix->rows + i]);
        }
        dd_debug("\n");
    }
}

void dd_info(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

void dd_fatal(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    exit(1);
}

double dd_timestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double timestamp = ts.tv_sec + ts.tv_nsec / 1e9;
    return timestamp;
}

void dd_check_cfitsio_status(int status)
{
    if (status)
    {
        fits_report_error(stderr, status);
        exit(status);
    }
}

typedef enum dd_status {dd_success, dd_error} dd_status;

// based on investigations with a hex editor and some
// example FITS files, NAXIS1 -> cols, NAXIS2 -> rows, NAXIS3 -> planes
// however the on-disk format (and the as-loaded format) are *not*
// column major! rows are stored contiguously, as are planes
dd_status dd_fits_to_doubles(char *input_file, double **image, long *cols, long *rows, long *planes) {
    fitsfile *infptr;
    int status = 0;
    fits_open_image(&infptr, input_file, READONLY, &status);
    dd_check_cfitsio_status(status);

    int naxis;
    fits_get_img_dim(infptr, &naxis, &status);
    dd_check_cfitsio_status(status);
    if (planes != NULL && naxis != 3) {
        fprintf(stderr, "Expected FITS cube %s to contain 3 axes, got %i\n", input_file, naxis);
        exit(1);
    } else if (planes == NULL && naxis != 2) {
        fprintf(stderr, "Expected FITS image %s to contain 2 axes, got %i\n", input_file, naxis);
        exit(1);
    }

    long image_dimensions[3];
    fits_get_img_size(infptr, naxis, /* cast to silence warning -> */ (long *)&image_dimensions, &status);
    dd_check_cfitsio_status(status);

    *cols = image_dimensions[0];
    *rows = image_dimensions[1];
    if (planes != NULL) {
        *planes = image_dimensions[2];
    }

    int total_n_elements = (*cols) * (*rows);
    if (planes != NULL) {
        total_n_elements *= (*planes);
    }
    *image = (double *)malloc(total_n_elements * sizeof(double));
    if (*image == NULL)
    {
        fprintf(stderr, "Unable to allocate RAM for image load\n");
        exit(1);
    }

    long first_pixel_indices[naxis];
    for (int i = 0; i < naxis; i++) {
        first_pixel_indices[i] = 1;
    }
    fits_read_pix(infptr, TDOUBLE, first_pixel_indices, total_n_elements,
                  0, *image, 0, &status);
    dd_check_cfitsio_status(status);
    fits_close_file(infptr, &status);
    dd_check_cfitsio_status(status);
    return dd_success;
}

dd_status dd_doubles_to_fits(char *output_file, double **image, long *cols, long *rows, long *planes) {
    fitsfile *outfptr;
    int status = 0;
    fits_create_diskfile(&outfptr, output_file, &status); // use _diskfile to avoid 'smart' filename parsing
    dd_check_cfitsio_status(status);
    int naxis = 1;
    long image_dimensions[3] = {0, 0, 0};
    long first_pixel_indices[3] = {1, 1, 1};
    dd_debug("%i \n", *cols);
    image_dimensions[0] = *cols;
    long long total_n_elements = (*cols);
    if (rows != NULL) {
        naxis = 2;
        image_dimensions[1] = *rows;
        total_n_elements *= (*rows);
    }
    if (planes != NULL) {
        naxis = 3;
        image_dimensions[2] = *planes;
        total_n_elements *= (*planes);
    }
    dd_debug("image_dimensions = %li x %li x %li\n", image_dimensions[0], image_dimensions[1], image_dimensions[2]);
    dd_debug("total_n_elements = %li\n", total_n_elements);
    fits_create_img(outfptr, DOUBLE_IMG, naxis, image_dimensions, &status);
    dd_check_cfitsio_status(status);
    dd_debug("after fits_create_img\n");
    fits_write_pix(outfptr, TDOUBLE, first_pixel_indices, total_n_elements, *image, &status);
    dd_check_cfitsio_status(status);
    dd_debug("after fits_write_pix\n");

    fits_close_file(outfptr, &status);
    dd_check_cfitsio_status(status);
    return dd_success;
}

dd_status dd_matrix_product(/* in */ double alpha,
                            /* in */ dd_Matrix *A,
                            /* in */ bool trans_A,
                            /* in */ dd_Matrix *B,
                            /* in */ bool trans_B,
                            /* in */ double beta,
                            /* in/out */ dd_Matrix *C) {
    int m, n, k;
    int lda, ldb;
    CBLAS_TRANSPOSE mod_A = CblasNoTrans;
    if (trans_A) {
        mod_A = CblasTrans;
        m = A->cols;
        k = A->rows;
        lda = k;
    } else {
        m = A->rows;
        k = A->cols;
        lda = m;
    }
    CBLAS_TRANSPOSE mod_B = CblasNoTrans;
    if (trans_B) {
        mod_B = CblasTrans;
        n = B->rows;
        ldb = n;
    } else {
        n = B->cols;
        ldb = k;
    }
    cblas_dgemm(
        CblasColMajor,
        mod_A,
        mod_B,
        m,
        n,
        k,
        alpha,
        A->data,
        lda,
        B->data,
        ldb,
        beta,
        C->data,
        n
    );
    return dd_success;
}

dd_status dd_subtract_mean_row(/* in */ dd_Matrix *matrix,
                                  /* out */ dd_Matrix *mean_sub_matrix,
                                  /* out */ double *column_means) {
    int cols = matrix->cols;
    int rows = matrix->rows;

    for (int c = 0; c < cols; c++) {
        column_means[c] = 0;
        for (int r = 0; r < rows; r++) {
            column_means[c] += matrix->data[c * rows + r];
        }
        column_means[c] = column_means[c] / (double)rows;
    }
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            mean_sub_matrix->data[c * rows + r] = matrix->data[c * rows + r] - column_means[c];
        }
    }
    return dd_success;
}

dd_status dd_subtract_mean_column(/* in */ dd_Matrix *matrix,
                               /* out */ dd_Matrix *mean_sub_matrix,
                               /* out */ double *row_means) {
    int cols = matrix->cols;
    int rows = matrix->rows;

    for (int r = 0; r < rows; r++) {
        row_means[r] = 0;
        for (int c = 0; c < cols; c++) {
            row_means[r] += matrix->data[c * rows + r];
        }
        row_means[r] = row_means[r] / (double)cols;
    }
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            mean_sub_matrix->data[c * rows + r] = matrix->data[c * rows + r] - row_means[r];
        }
    }
    return dd_success;
}

dd_status dd_sample_covariance(/* in */ dd_Matrix *matrix,
                               /* out */ dd_Matrix *cov_matrix) {
    double *row_means = (double *)malloc(matrix->rows * sizeof(double));
    double *mean_sub_data = (double *)malloc(matrix->rows * matrix->cols * sizeof(double));
    dd_Matrix mean_sub_matrix = {
        .rows = matrix->rows,
        .cols = matrix->cols,
        .data = mean_sub_data
    };
    dd_subtract_mean_column(matrix, &mean_sub_matrix, row_means);

    float prefactor = 1.0 / (float)(matrix->cols - 1);

    dd_matrix_product(prefactor, &mean_sub_matrix, false, &mean_sub_matrix, true, 0.0, cov_matrix);
    free(mean_sub_data);
    return dd_success;
}

dd_status dd_make_transpose(double *input_matrix, double **output_matrix, long cols, long rows) {
    long total_n_elements = rows * cols;
    *output_matrix = (double *)malloc(total_n_elements * sizeof(double));
    if (*output_matrix == NULL) {
        exit(1);
    }
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            (*output_matrix)[c * rows + r] = input_matrix[r * cols + c];
        }
    }
    return dd_success;
}

typedef struct dd_IntWorkspace {
    lapack_int length;
    lapack_int *data;
} dd_IntWorkspace;

typedef struct dd_DoubleWorkspace {
    int length;
    double *data;
} dd_DoubleWorkspace;

char DD_COMPUTE_EVALS = 'N';
char DD_COMPUTE_EVECS = 'V';
char DD_ALL_EVALS = 'A';         // 'A': all eigenvalues will be found.
char DD_EVALS_VALUE_RANGE = 'V'; // 'V': all eigenvalues in the half-open interval (VL,VU] will be found.
char DD_EVALS_INDEX_RANGE = 'I'; // 'I': the IL-th through IU-th eigenvalues will be found.
char DD_UPPER_TRIANGULAR = 'U';
char DD_LOWER_TRIANGULAR = 'L';


int _comparator_doubles(const void *lhs, const void *rhs) {
    double left = *(const double *)lhs;
    double right = *(const double *)rhs;
    if (left < right) {
        return -1;
    }
    if (left > right) {
        return 1;
    }
    return 0;
}

bool dd_doubles_all_close(double *a, double *b, long length, double tol) {
    for (size_t i = 0; i < length; i++) {
        if (fabs(a[i] - b[i]) > tol) {
            return false;
        }
    }
    return true;
}

dd_status dd_sort_doubles(double *a, size_t length) {
    qsort(a, length, sizeof(double), _comparator_doubles);
    return dd_success;
}

/*
 * Compute eigenvectors and eigenvalues with MKL dsyevr
 */
dd_status dd_mkl_syevr(/* in */ dd_Matrix *matrix,
                       /* out */ dd_Matrix *eigenvectors,
                       /* out */ dd_Matrix *eigenvalues,
                       /* out */ double *time_elapsed,
                       /* optional: */ dd_DoubleWorkspace **workspace_ptrptr,
                       /* optional: */ dd_IntWorkspace **intworkspace_ptrptr) {
    if ((*workspace_ptrptr == NULL && *intworkspace_ptrptr != NULL) || (*workspace_ptrptr != NULL && *intworkspace_ptrptr == NULL)) {
        dd_info("If you supply 'work', you must supply 'iwork' and vice versa\n");
        return dd_error;
    }
    MKL_INT info;
    MKL_INT n = matrix->cols;
    MKL_INT lda = matrix->rows;
    // The support of the eigenvectors in Z, i.e., the indices
    // indicating the nonzero elements in Z.
    // (Not used since all evecs should be != 0)
    MKL_INT _isuppz_unused[2 * n];
    MKL_INT il = matrix->cols - eigenvectors->cols + 1, iu = matrix->cols;
    dd_debug(
        "'eigenvectors' is %i x %i, 'eigenvalues' is %i x %i\n",
        eigenvectors->rows,
        eigenvectors->cols,
        eigenvalues->rows,
        eigenvalues->cols
    );
    dd_debug("il = %li, iu = %li\n", il, iu);
    if (eigenvectors->cols != eigenvalues->cols) {
        return dd_error;
    }

    if (eigenvectors->data == NULL) {
        dd_info("null pointer in eigenvectors struct\n");
        exit(1);
    }
    if (eigenvalues->data == NULL) {
        dd_info("null pointer in eigenvalues struct\n");
        exit(1);
    }
    MKL_INT m; // num eigenvectors found (filled by lapack)
    double abstol = -1; // use default tolerance
    double _vl_unused = 0, _vu_unused = 0; // "Not referenced if RANGE = 'A' or 'I'."
    if (*workspace_ptrptr == NULL) {
        *workspace_ptrptr = (dd_DoubleWorkspace*)malloc(sizeof(dd_DoubleWorkspace));
        *intworkspace_ptrptr = (dd_IntWorkspace*)malloc(sizeof(dd_IntWorkspace));
        double work_query;
        MKL_INT iwork_query;
        MKL_INT lwork_query = -1, liwork_query = -1;
        dd_debug("Workspace query iu = %li\n", iu);
        info = LAPACKE_dsyevr_work(
                LAPACK_COL_MAJOR, // matrix_layout
                DD_COMPUTE_EVECS, // jobz
                // DD_EVALS_INDEX_RANGE, // range
                DD_ALL_EVALS, // range
                DD_UPPER_TRIANGULAR, // uplo
                matrix->cols, // n
                matrix->data, // a
                matrix->rows, // lda
                _vl_unused,
                _vu_unused,
                il,
                iu,
                abstol,
                &m, //
                eigenvalues->data,
                eigenvectors->data,
                eigenvectors->rows,
                _isuppz_unused,
                &work_query,
                lwork_query,
                &iwork_query,
                liwork_query
        );
        if( info != 0 ) {
            dd_fatal("Workspace query failed\n");
        }
        (*workspace_ptrptr)->length = (MKL_INT)work_query;
        (*workspace_ptrptr)->data = (double*)mkl_malloc((*workspace_ptrptr)->length * sizeof(double), 64);
        if ((*workspace_ptrptr)->data == NULL) {
            dd_fatal("malloc returned NULL!\n");
        }
        dd_debug("Successfully allocated double workspace of length %i\n", (*workspace_ptrptr)->length);
        (*intworkspace_ptrptr)->length = (lapack_int)iwork_query;
        (*intworkspace_ptrptr)->data = (lapack_int*)mkl_malloc((*intworkspace_ptrptr)->length * sizeof(lapack_int), 64);
        dd_debug("Successfully allocated int workspace of length %i\n", (*intworkspace_ptrptr)->length);
    }
    dd_DoubleWorkspace workspace = **workspace_ptrptr;
    dd_IntWorkspace intworkspace = **intworkspace_ptrptr;

    *time_elapsed = dd_timestamp();
    info = LAPACKE_dsyevr_work(
        LAPACK_COL_MAJOR,
        DD_COMPUTE_EVECS,
        DD_ALL_EVALS, // range
        DD_UPPER_TRIANGULAR,
        matrix->cols,
        matrix->data,
        matrix->rows,
        _vl_unused, _vu_unused,
        il,
        iu,
        abstol,
        &m,
        eigenvalues->data,
        eigenvectors->data,
        eigenvectors->rows,
        _isuppz_unused,
        workspace.data,
        workspace.length,
        intworkspace.data,
        intworkspace.length
    );
    if( info != 0 ) {
        dd_fatal("MKL dsyevr failed\n");
    }
    *time_elapsed = -(*time_elapsed - dd_timestamp());
    // dd_info("Took %f for mkl_syevr\n", *time_elapsed);
    return dd_success;
}
#endif