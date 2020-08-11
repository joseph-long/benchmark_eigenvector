#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <strings.h>
#include <errno.h>
#include "fitsio.h"
// #define DD_DEBUG
#include "doodads.h"
#include "mkl.h"

// don't want to time covariance calc, use a global
// to persist it outside work func
// static double *precalculated_cov;
// static double *image;
static long cols, rows, planes;
static int cube_flag = 0;
static int ramp_increment = -1;
static long number_of_images = 0;
static long number_of_eigenvectors = 0;
static int warmups = 1;
static int iterations = 1;

static char HEADER[] = "rows\tcols\tmethod\tn_evecs\ttime_spent\n";
static char FORMAT[] = "%li\t%li\t%s\t%li\t%e\n";

static struct option long_options[] = {
    {"cube", no_argument, NULL, 'c'},
    {"number_of_images", required_argument, NULL, 'n'},
    {"number_of_eigenvectors", required_argument, NULL, 'n'},
    {"method", required_argument, NULL, 'm'},
    {"warmups", required_argument, NULL, 'w'},
    {"iterations", required_argument, NULL, 'i'},
    {"ramp", required_argument, NULL, 'r'},
    {"outprefix", required_argument, NULL, 'o'},
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

char *construct_filename(char *dest, char *prefix, char *suffix) {
    // printf("%x %x %x\n", dest, prefix, suffix);
    strcpy(dest, "");
    strcat(dest, prefix);
    strcat(dest, suffix);
    return dest;
}

void remove_if_exists(char *filepath) {
    if (remove(filepath) != 0) {
        if (errno != ENOENT) {
            dd_debug("Error: %s", strerror(errno));
            exit(1);
        }
    }
}

int main(int argc, char *argv[])
{
    char option_code;
    char outprefix[500] = "";
    while ((option_code = getopt_long(argc, argv, ":cn:e:m:w:i:r:o:", long_options, NULL)) != -1)
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
            strcat(outprefix, optarg);
            break;
        case 'r':
            ramp_increment = strtol(optarg, NULL, 10);
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
    dd_debug("Output file prefix (-o OUTPREFIX): %s\n", outprefix);
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
    char output_file[512]; // dest buffer for construcitng cov, evec, eval output names
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
        double *covdata = data_matrix.data;
        long rows, cols;
        rows = data_matrix.rows;
        cols = data_matrix.cols;
        if (strlen(outprefix) != 0) {
            construct_filename(output_file, outprefix, "_cov.fits");
            remove_if_exists(output_file);
            dd_doubles_to_fits(output_file, &covdata, &cols, &rows, NULL);
            dd_debug("Saving covariance to %s\n", output_file);
        }
    } else {
        data_matrix = input_matrix;
    }

    // make space for a copy for the solver to chew up
    size_t data_size = data_matrix.cols * data_matrix.rows * sizeof(double);
    double *pristine_data = (double *)malloc(data_size);
    memcpy(pristine_data, data_matrix.data, data_size);

    // Handle default number of evecs
    if (number_of_eigenvectors == 0) {
        number_of_eigenvectors = data_matrix.cols;
    }

    // make space for evecs output
    // double *output = (double *)malloc(num_evecs * rows * sizeof(double));
    dd_Matrix out_eigenvectors = {
        .cols = number_of_eigenvectors,
        .rows = data_matrix.rows,
        .data = (double *)calloc(number_of_eigenvectors * data_matrix.rows, sizeof(double))
    };
    dd_Matrix out_eigenvalues = {
        .cols = number_of_eigenvectors,
        .rows = 1,
        .data = (double *)calloc(number_of_eigenvectors, sizeof(double))
    };
    double out_time_elapsed;
    dd_DoubleWorkspace *inout_workspace = NULL;
    dd_IntWorkspace *inout_intworkspace = NULL;
    if (strlen(outprefix) != 0 && warmups == 0) {
        // force one warm-up to have data to write
        warmups = 1;
    }
    // warmup iterations
    for (int i = 0; i < warmups; i++) {
        // restore pristine data (yes this appears to be needed)
        memcpy(data_matrix.data, pristine_data, data_size);
        dd_mkl_syevr(&data_matrix, &out_eigenvectors, &out_eigenvalues, &out_time_elapsed, &inout_workspace, &inout_intworkspace);
    }
    // timer iterations
    if (iterations > 0) {
        printf(HEADER);
        double start, end, total_time = 0; // timestamp values
        for (int i = 0; i < iterations; i++) {
            
            if (ramp_increment > 0) {
                int current_n_evecs = 1;
                while (1) {
                    dd_debug("ramp %i / %i\n", current_n_evecs, number_of_eigenvectors);
                    out_eigenvectors.cols = current_n_evecs;
                    out_eigenvalues.cols = current_n_evecs;
                    // restore pristine data (yes this appears to be needed)
                    memcpy(data_matrix.data, pristine_data, data_size);
                    start = dd_timestamp();
                    dd_mkl_syevr(
                        &data_matrix,
                        &out_eigenvectors,
                        &out_eigenvalues,
                        &out_time_elapsed,
                        &inout_workspace,
                        &inout_intworkspace
                    );
                    end = dd_timestamp();
                    total_time += end - start;
                    double time_spent = end - start;
                    // rows cols method n_evecs time_spent
                    printf(FORMAT,
                        data_matrix.rows,
                        data_matrix.cols,
                        string_from_method(method),
                        out_eigenvectors.cols,
                        time_spent
                    );
                    if (current_n_evecs == number_of_eigenvectors) {
                        break;
                    }
                    current_n_evecs += ramp_increment;
                    if (current_n_evecs > number_of_eigenvectors) {
                        // overshot the upper bound
                        current_n_evecs = number_of_eigenvectors;
                    }
                }
            } else {
                // restore pristine data (yes this appears to be needed)
                memcpy(data_matrix.data, pristine_data, data_size);
                start = dd_timestamp();
                dd_mkl_syevr(
                    &data_matrix,
                    &out_eigenvectors,
                    &out_eigenvalues,
                    &out_time_elapsed,
                    &inout_workspace,
                    &inout_intworkspace
                );
                end = dd_timestamp();
                total_time += end - start;
                double time_spent = end - start;
                // rows cols method n_evecs time_spent
                printf(FORMAT,
                    data_matrix.rows,
                    data_matrix.cols,
                    string_from_method(method),
                    out_eigenvectors.cols,
                    time_spent
                );
            }
            
            dd_debug("%f sec\n", end - start, end, start);
        }
        // dd_print_matrix(&out_eigenvalues);

        // write timing info to stdout
        // dd_debug("%i iterations, avg %f sec each\n", iterations, total_time / iterations);
    }
    // optionally write evecs to fits
    if (strlen(outprefix) != 0) {
        // Save eigenvectors
        double *transposed_evecs;
        dd_make_transpose(out_eigenvectors.data, &transposed_evecs, out_eigenvectors.cols, out_eigenvectors.rows);
        dd_debug("Made transpose %x\n", transposed_evecs);
        long cols, rows;
        cols = out_eigenvectors.cols;
        rows = out_eigenvectors.rows;
        construct_filename(output_file, outprefix, "_evecs.fits");
        remove_if_exists(output_file);
        dd_debug("Saving eigenvectors to %s\n", output_file);
        dd_doubles_to_fits(output_file, &transposed_evecs, &cols, &rows, NULL);

        // Save eigenvalues
        construct_filename(output_file, outprefix, "_evals.fits");
        remove_if_exists(output_file);
        dd_debug("Saving eigenvalues to %s\n", output_file);
        rows = 1;
        dd_doubles_to_fits(output_file, &out_eigenvalues.data, &cols, NULL, NULL);
    }
    return 0;
}