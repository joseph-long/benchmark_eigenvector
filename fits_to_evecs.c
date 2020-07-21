#include <stdio.h>
#include <getopt.h>
#include <strings.h>
#include "fitsio.h"
#include "doodads.h"
#include "mkl.h"

// don't want to time covariance calc, use a global
// to persist it outside work func
static double *precalculated_cov;
static double *image;
static long cols, rows, planes;
static int cube_flag = 0;
static long number_of_images = 0;
static long number_of_eigenvectors = 0;
static int warmups = 1;
static int iterations = 1;

static struct option long_options[] = {
    {"cube", no_argument, NULL, 'c'},
    {"number_of_images", required_argument, NULL, 'n'},
    {"method", required_argument, NULL, 'm'},
    {"warmups", required_argument, NULL, 'w'},
    {"iterations", required_argument, NULL, 'i'},
};

enum method_choices
{
    mkl_syevr
};
static inline const char *string_from_method(enum method_choices val)
{
    static const char *strings[] = {
        "mkl_syevr"};

    return strings[val];
}

static enum method_choices method = mkl_syevr;

// Method functions

/* Auxiliary routine: printing a matrix */
void print_matrix(char *desc, MKL_INT m, MKL_INT n, double *a, MKL_INT lda)
{
    MKL_INT i, j;
    printf("\n %s\n", desc);
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            printf(" %6.2f", a[i * lda + j]);
        printf("\n");
    }
}

char compute_evals_only = 'N';
char compute_evecs_too = 'V';
char select_all = 'A';
char select_by_value = 'V';
char select_by_indices = 'I';
char upper_triangular = 'U';
char lower_triangular = 'L';

void do_mkl_syevr_colmaj(double *image, double *output, long rows, long cols, long num_evecs) {
    if (rows != cols) {
        fprintf(stderr, "Non-square matrix (got %li rows x %li cols)\n", rows, cols);
        exit(1);
    }
    MKL_INT n = cols, il, iu, m, lda = rows, ldz = cols, info;
    double abstol, vl, vu;
    MKL_INT isuppz[cols];
    double w[cols], z[ldz * num_evecs];
    abstol = -1.0;
    il = 1;
    iu = num_evecs;
    info = LAPACKE_dsyevr(
        LAPACK_COL_MAJOR,
        compute_evecs_too,
        select_by_indices,
        upper_triangular,
        n,
        image,
        lda,
        vl, vu,
        il, iu,
        abstol,
        &m,
        w,
        z,
        ldz,
        isuppz);
    /* Check for convergence */
    if (info > 0)
    {
        printf("The algorithm failed to compute eigenvalues.\n");
        exit(1);
    }
    /* Print the number of eigenvalues found */
    printf("\n The total number of eigenvalues found:%2i\n", m);
    /* Print eigenvalues */
    print_matrix("Selected eigenvalues", 1, m, w, 1);
    /* Print eigenvectors */
    print_matrix("Selected eigenvectors (stored columnwise)", n, m, z, ldz);
}

void do_mkl_syevr_rowmaj(double *image, double *output, long rows, long cols, long num_evecs)
{
    // int N = 5;
    // int NSELECT = 3;
    // int LDA = N;
    // int LDZ = NSELECT;
    /* Locals */
    MKL_INT n = rows, il, iu, m, lda = cols, ldz = num_evecs, info;
    double abstol, vl, vu;
    /* Local arrays */
    MKL_INT isuppz[rows];
    double w[rows], z[num_evecs * rows];
    // double a[LDA*N] = {
    // 0.67, -0.20, 0.19, -1.06, 0.46,
    // 0.00,  3.82, -0.13,  1.06, -0.48,
    // 0.00,  0.00, 3.27,  0.11, 1.10,
    // 0.00,  0.00, 0.00,  5.86, -0.98,
    // 0.00,  0.00, 0.00,  0.00, 3.54
    // };
    /* Executable statements */
    printf("LAPACKE_dsyevr (row-major, high-level) Example Program Results\n");
    /* Negative abstol means using the default value */
    abstol = -1.0;
    /* Set il, iu to compute NSELECT smallest eigenvalues */
    il = 1;
    iu = num_evecs;
    /* Solve eigenproblem */
    info = LAPACKE_dsyevr(
        LAPACK_ROW_MAJOR,
        compute_evecs_too,
        select_by_indices,
        upper_triangular,
        n,
        image,
        lda,
        vl, vu,
        il, iu,
        abstol,
        &m,
        w,
        z,
        ldz,
        isuppz);
    /* Check for convergence */
    if (info > 0)
    {
        printf("The algorithm failed to compute eigenvalues.\n");
        exit(1);
    }
    /* Print the number of eigenvalues found */
    printf("\n The total number of eigenvalues found:%2i\n", m);
    /* Print eigenvalues */
    print_matrix("Selected eigenvalues", 1, m, w, 1);
    /* Print eigenvectors */
    print_matrix("Selected eigenvectors (stored columnwise)", n, m, z, ldz);
}

// end of work functions

int main(int argc, char *argv[])
{
    char option_code;
    while ((option_code = getopt_long(argc, argv, ":cn:e:m:w:i:", long_options, NULL)) != -1)
    {
        switch (option_code)
        {
        case 'c':
            cube_flag = 1;
            break;
        case 'n':
            number_of_images = strtol(optarg, NULL, 10);
            break;
        case 'e':
            number_of_eigenvectors = strtol(optarg, NULL, 10);
            break;
        case 'm':
            if (strcasecmp(optarg, "mkl_syevr") == 0)
            {
                method = mkl_syevr;
            }
            else
            {
                fprintf(stderr, "%s: method must be one of: mkl_syevr (got %s)\n", argv[0], optarg);
                exit(1);
            }
            break;
        case 'w':
            warmups = strtol(optarg, NULL, 10);
            break;
        case 'i':
            iterations = strtol(optarg, NULL, 10);
            break;
        case ':':
            /* missing option argument */
            fprintf(stderr, "%s: option '-%c' requires an argument\n",
                    argv[0], optopt);
            exit(1);
            break;
        case '?':
        default:
            /* invalid option */
            fprintf(stderr, "%s: option '-%c' is invalid: ignored\n",
                    argv[0], optopt);
            exit(1);
            break;
        }
    }
    printf("Load as cube? %i\n", cube_flag);
    printf("Number of images: %li (0 means use whole file)\n", number_of_images);
    printf("Number of eigenvectors: %li (0 means use same as num. images)\n", number_of_eigenvectors);
    printf("Method: %s\n", string_from_method(method));
    printf("Warmups: %i\n", warmups);
    printf("Iterations: %i\n", iterations);
    for (int i = optind; i < argc; i++)
    {
        printf("%s\n", argv[i]);
    }
    if (argc == optind)
    {
        fprintf(stderr, "%s: Must supply input FITS file\n", argv[0]);
        exit(1);
    }
    char *input_file = argv[optind];
    char *output_file;
    if (optind + 1 < argc)
    {
        output_file = argv[optind + 1];
    }
    // load fits cube
    double *image;
    long cols, rows, planes;
    long *optional_planes_ptr;
    // Should loader expect cube?
    if (cube_flag)
    {
        optional_planes_ptr = &planes;
    }
    else
    {
        optional_planes_ptr = NULL;
    }
    dd_fits_to_doubles(input_file, &image, &cols, &rows, optional_planes_ptr);
    printf("Loaded %s:\n", input_file);
    if (cube_flag)
    {
        printf("\t%li planes\n", planes);
    }
    printf("\t%li rows\n", rows);
    printf("\t%li columns\n", cols);

    long mtx_rows, mtx_cols;
    if (cube_flag)
    {
        mtx_cols = rows * cols;
        mtx_rows = planes;
    }
    else
    {
        mtx_cols = cols;
        mtx_rows = rows;
    }
    // TODO this as an arg
    int num_evecs = 3;
    // Transpose to column-major for LAPACK reasons
    double *transposed_image;
    dd_make_transpose(image, &transposed_image, mtx_cols, mtx_rows);
    size_t data_size = mtx_cols * mtx_rows * sizeof(double);

    // if method requires square covariance matrix as input,
    // compute it:
    double *data_matrix;
    if (method == mkl_syevr) {
        
    } else {
        data_matrix = transposed_image;
    }


    // make space for evecs output
    double *output = (double *)malloc(num_evecs * rows * sizeof(double));
    // make space for a copy for the solver to chew up
    double *matrix_to_process = (double *)malloc(data_size);
    // warmup iterations
    for (int i = 0; i < warmups; i++) {
        memcpy(matrix_to_process, transposed_image, data_size);
        do_mkl_syevr_colmaj(matrix_to_process, output, rows, cols, num_evecs);
    }
    // timer iterations
    double start, end, total_time; // timestamp values
    for (int i = 0; i < iterations; i++) {
        memcpy(matrix_to_process, transposed_image, data_size);
        start = dd_timestamp();
        do_mkl_syevr_colmaj(matrix_to_process, output, rows, cols, num_evecs);
        end = dd_timestamp();
        total_time += end - start;
        printf("%f sec\n", end - start, end, start);
    }
    // optionally write evecs to fits
    // write timing info to stdout
    printf("%i iterations, avg %f sec each\n", iterations, total_time / iterations);
    return 0;
}