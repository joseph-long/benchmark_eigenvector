#include <stdio.h>
#include <getopt.h>
#include <strings.h>
#include "fitsio.h"
#define DD_DEBUG
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
    {"outfile", required_argument, NULL, 'o'},
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

int main(int argc, char *argv[])
{
    char option_code;
    char *outfile = NULL;
    while ((option_code = getopt_long(argc, argv, ":cn:e:m:w:i:o:", long_options, NULL)) != -1)
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
        case 'o':
            outfile = optarg;
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
    dd_debug("Load as cube (-c)? %i\n", cube_flag);
    dd_debug("Number of images (-n NUM): %li (0 means use whole file)\n", number_of_images);
    dd_debug("Number of eigenvectors (-e NUM): %li (0 means use same as num. images)\n", number_of_eigenvectors);
    dd_debug("Method (-m METHOD): %s\n", string_from_method(method));
    dd_debug("Warmups (-w NUM): %i\n", warmups);
    dd_debug("Iterations (-i NUM): %i\n", iterations);
    dd_debug("Output file path (-o OUTFILE): %s\n", outfile);
    for (int i = optind; i < argc; i++)
    {
        dd_debug("%i %s\n", i, argv[i]);
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
    dd_debug("Loaded %s:\n", input_file);
    if (cube_flag)
    {
        dd_debug("\t%li planes\n", planes);
    }
    dd_debug("\t%li rows\n", rows);
    dd_debug("\t%li columns\n", cols);
    // Handle default number of images
    if (number_of_images == 0) {
        dd_debug("Using all images: ");
        if (cube_flag) {
            number_of_images = planes;
        } else {
            number_of_images = cols;
        }
        dd_debug("number_of_images = %i\n", number_of_images);
    }

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
    // int num_evecs = 3;
    // Transpose to column-major for LAPACK reasons
    double *transposed_image;
    dd_make_transpose(image, &transposed_image, mtx_cols, mtx_rows);
    size_t data_size = mtx_cols * mtx_rows * sizeof(double);
    // wrap transposed in dd_Matrix struct
    dd_Matrix input_matrix = {
        .cols = mtx_cols,
        .rows = mtx_rows,
        .data = transposed_image
    };
    // if method requires square covariance matrix as input,
    // compute it:
    dd_Matrix data_matrix;
    if (method == mkl_syevr) {
        data_matrix.rows = mtx_rows;
        data_matrix.cols = mtx_rows;
        data_matrix.data = (double *)malloc(data_matrix.rows * data_matrix.cols * sizeof(double));
        dd_sample_covariance(&input_matrix, &data_matrix);
        dd_debug("cov mtx %i x %i\n", data_matrix.rows, data_matrix.cols);
    } else {
        data_matrix = input_matrix;
    }

    // Handle default number of evecs
    if (number_of_eigenvectors == 0) {
        number_of_eigenvectors = data_matrix.cols;
    }

    // make space for evecs output
    // double *output = (double *)malloc(num_evecs * rows * sizeof(double));
    // make space for a copy for the solver to chew up
    // double *matrix_to_process = (double *)malloc(data_size);
    dd_Matrix out_eigenvectors = {
        .cols = number_of_eigenvectors,
        .rows = data_matrix.rows,
        .data = (double *)malloc(number_of_eigenvectors * data_matrix.rows * sizeof(double))
    };
    dd_Matrix out_eigenvalues = {
        .cols = number_of_eigenvectors,
        .rows = 1,
        .data = (double *)malloc(number_of_eigenvectors * sizeof(double))
    };
    double out_time_elapsed;
    dd_DoubleWorkspace inout_workspace = {
      .length = 0,
      .data = NULL
    };
    dd_IntWorkspace inout_intworkspace = {
      .length = 0,
      .data = NULL
    };
    // warmup iterations
    for (int i = 0; i < warmups; i++) {
        dd_mkl_syevr(&data_matrix, &out_eigenvectors, &out_eigenvalues, &out_time_elapsed, &inout_workspace, &inout_intworkspace);
    }
    // timer iterations
    double start, end, total_time; // timestamp values
    for (int i = 0; i < iterations; i++) {
        start = dd_timestamp();
        dd_mkl_syevr(&data_matrix, &out_eigenvectors, &out_eigenvalues, &out_time_elapsed, &inout_workspace, &inout_intworkspace);
        end = dd_timestamp();
        total_time += end - start;
        dd_debug("%f sec\n", end - start, end, start);
    }
    dd_print_matrix(&out_eigenvectors);
    // optionally write evecs to fits
    if (outfile != NULL) {
        double *transposed_image;
        dd_make_transpose(out_eigenvectors.data, &transposed_image, out_eigenvectors.cols, out_eigenvectors.rows);
        dd_debug("Made transpose %x\n", transposed_image);
        long cols, rows;
        cols = out_eigenvectors.cols;
        rows = out_eigenvectors.rows;
        dd_doubles_to_fits(outfile, transposed_image, &cols, &rows, NULL);
    }
    // write timing info to stdout
    dd_debug("%i iterations, avg %f sec each\n", iterations, total_time / iterations);
    return 0;
}