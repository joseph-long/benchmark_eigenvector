#include <stdbool.h>
#include <math.h>
#include "stdio.h"
#include "greatest.h"
#include "../doodads.h"

TEST load_fits_image_float(void)
{
    double *image;
    long cols, rows;
    dd_fits_to_doubles(
        "./data/indices_16x4_float.fits",
        &image,
        &cols,
        &rows, NULL);
    ASSERT_EQ_FMT(16l, rows, "%li");
    ASSERT_EQ_FMT(4l, cols, "%li");
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            dd_debug("[row %i, col %i, idx %i] = %f \n", row, col, row * cols + col, image[row * cols + col]);
            ASSERT_EQ(row * cols + col, image[row * cols + col]);
        }
    }
    PASS();
}

TEST load_fits_image_int_conversion(void)
{
    double *image;
    long cols, rows;
    dd_fits_to_doubles(
        "./data/indices_16x4.fits",
        &image,
        &cols,
        &rows, NULL);
    ASSERT_EQ_FMT(16l, rows, "%li");
    ASSERT_EQ_FMT(4l, cols, "%li");
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            dd_debug("[row %i, col %i, idx %i] = %f \n", row, col, row * cols + col, image[row * cols + col]);
            ASSERT_EQ(row * cols + col, image[row * cols + col]);
        }
    }
    PASS();
}

TEST load_fits_cube(void)
{
    double *image;
    long cols, rows, planes;
    dd_fits_to_doubles(
        "./data/sequential_cube_2cols_3rows_4planes.fits",
        &image,
        &cols,
        &rows,
        &planes);
    ASSERT_EQ_FMT(2l, cols, "%li");
    ASSERT_EQ_FMT(3l, rows, "%li");
    ASSERT_EQ_FMT(4l, planes, "%li");
    int elements_per_plane = rows * cols;
    for (int plane = 0; plane < planes; plane++)
    {
        for (int i = 0; i < elements_per_plane; i++)
        {
            ASSERT_EQ_FMT((double)plane + 1, image[plane * elements_per_plane + i], "%f");
        }
    }
    PASS();
}

TEST roundtrip_fits_cube(void)
{
    double *image;
    long cols, rows, planes;
    dd_fits_to_doubles(
        "./data/sequential_cube_2cols_3rows_4planes.fits",
        &image,
        &cols,
        &rows,
        &planes);
    ASSERT_EQ_FMT(2l, cols, "%li");
    ASSERT_EQ_FMT(3l, rows, "%li");
    ASSERT_EQ_FMT(4l, planes, "%li");
    int elements_per_plane = rows * cols;

    char outfile[1024];
    snprintf(outfile, 1024, "/tmp/test_roundtrip_%f.fits", dd_timestamp());

    dd_doubles_to_fits(outfile, &image, &cols, &rows, &planes);

    double *image2;
    long cols2, rows2, planes2;
    dd_fits_to_doubles(
        "./data/sequential_cube_2cols_3rows_4planes.fits",
        &image2,
        &cols2,
        &rows2,
        &planes2);

    // Check nothing was changed by writing and reading back
    ASSERT_EQ_FMT(cols, cols2, "%li");
    ASSERT_EQ_FMT(rows, rows2, "%li");
    ASSERT_EQ_FMT(planes, planes2, "%li");

    for (int plane = 0; plane < planes; plane++)
    {
        for (int i = 0; i < elements_per_plane; i++)
        {
            ASSERT_EQ_FMT(image[plane * elements_per_plane + i], image2[plane * elements_per_plane + i], "%f");
        }
    }
    int status = remove(outfile);
    ASSERT_EQm("Unable to delete temp file", status, 0);
    PASS();
}

TEST make_transpose(void) {
    long rows = 3;
    long cols = 4;
    // double row_major_mtx[3 * 4] = {
    //     1, 1, 1, 1,
    //     2, 2, 2, 2,
    //     3, 3, 3, 3,
    // };
    // double col_major_mtx[4 * 3] = {
    //     1, 2, 3,
    //     1, 2, 3,
    //     1, 2, 3,
    //     1, 2, 3,
    // };
    double row_major_mtx[3 * 4] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    };
    double col_major_mtx[4 * 3] = {
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12,
    };
    double *transpose;
    dd_convert_to_colmajor(row_major_mtx, &transpose, cols, rows);
    for (long idx = 0; idx < 12; idx++) {
        ASSERT_EQ(transpose[idx], col_major_mtx[idx]);
    }
    PASS();
}

TEST get_eigenvectors_mkl_syevr(void) {
    int N = 5;
    // int NSELECT = 3;
    int LDA = N;
    double a[5*5] = {
        0.67,  0.00,  0.00,  0.00,  0.00,
        -0.20,  3.82,  0.00,  0.00,  0.00,
        0.19, -0.13,  3.27,  0.00,  0.00,
        -1.06,  1.06,  0.11,  5.86,  0.00,
        0.46, -0.48,  1.10, -0.98,  3.54
    };
    double eigenvectors[5 * 5] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    double eigenvalues[5] = {0, 0, 0, 0, 0};
    dd_Matrix my_matrix = {
        .rows = N,
        .cols = LDA,
        .data = (double *)&a
    };
    dd_Matrix my_eigenvectors = {
        .rows = N,
        .cols = LDA,
        .data = (double *)&eigenvectors
    };
    dd_Matrix my_eigenvalues = {
        .rows = 1,
        .cols = N,
        .data = (double *)&eigenvalues
    };
    double duration;
    dd_IntWorkspace *my_intworkspace = NULL;
    dd_DoubleWorkspace *my_workspace = NULL;
    dd_status status;
    status = dd_mkl_syevr(
        &my_matrix, &my_eigenvectors, &my_eigenvalues, &duration,
        &my_workspace, &my_intworkspace
    );
    dd_sort_doubles(my_eigenvalues.data, my_eigenvalues.rows);
    ASSERT_EQm("dd_mkl_syevr didn't return dd_success", status, dd_success);
    // double x[2] = {1,2};
    double known_eigenvalues[5] = {
        0.43302179880852787,
        2.144946655568288,
        3.3680867378650334,
        4.279153022898037,
        6.934791784860096
    };
    ASSERTm("Eigenvalues don't match reference", dd_doubles_all_close(my_eigenvalues.data, known_eigenvalues, 5, 1e-13));
    dd_IntWorkspace *iwptr = my_intworkspace;
    dd_DoubleWorkspace *wptr = my_workspace;
    status = dd_mkl_syevr(
        &my_matrix, &my_eigenvectors, &my_eigenvalues, &duration,
        &my_workspace, &my_intworkspace
    );
    ASSERT_EQm("dd_mkl_syevr didn't return dd_success", status, dd_success);
    ASSERT_EQm("Got new workspace on second invocation (not reusing!)", my_workspace, wptr);
    ASSERT_EQm("Got new intworkspace on second invocation (not reusing!)", my_intworkspace, iwptr);
    PASS();
}

TEST get_eigenvector_range_mkl_syevr(void) {
    int N = 5;
    int LDA = N;
    int number_evals = 3;
    double a[5*5] = {
        0.67,  0.00,  0.00,  0.00,  0.00,
        -0.20,  3.82,  0.00,  0.00,  0.00,
        0.19, -0.13,  3.27,  0.00,  0.00,
        -1.06,  1.06,  0.11,  5.86,  0.00,
        0.46, -0.48,  1.10, -0.98,  3.54
    };
    // three columns vectors, 5 entries per vector
    double eigenvectors[3 * 5] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    double eigenvalues[5] = {0, 0, 0};
    dd_Matrix my_matrix = {
        .rows = N,
        .cols = LDA,
        .data = (double *)&a
    };
    dd_Matrix my_eigenvectors = {
        .rows = N,
        .cols = number_evals,
        .data = (double *)&eigenvectors
    };
    dd_Matrix my_eigenvalues = {
        .rows = 1,
        .cols = number_evals,
        .data = (double *)&eigenvalues
    };
    double duration;
    dd_IntWorkspace *my_intworkspace = NULL;
    dd_DoubleWorkspace *my_workspace = NULL;
    dd_status status;
    status = dd_mkl_syevr(
        &my_matrix, &my_eigenvectors, &my_eigenvalues, &duration,
        &my_workspace, &my_intworkspace
    );
    ASSERT_EQm("dd_mkl_syevr didn't return dd_success", status, dd_success);
    double known_eigenvalues[3] = {
        3.3680867378650334,
        4.279153022898037,
        6.934791784860096
    };
    double known_eigenvectors[3 * 5] = {
        -0.08176685, -0.93380532, -0.07351991,  0.31294971, -0.13408617, // eval 3.36...
        -0.01241397,  0.02572566, -0.70867609, -0.35413453, -0.60954985, // eval 4.27...
        0.18243522, -0.35606835,  0.10215486, -0.84049795,  0.35079952, // eval 6.93...
    };
    ASSERTm("Eigenvalues don't match reference", dd_doubles_all_close(my_eigenvalues.data, known_eigenvalues, 5, 1e-13));
    for (int i = 0; i < number_evals; i++) {
        bool found = false;
        int comparedto;
        for (comparedto = 0; comparedto < number_evals; comparedto++) {
            dd_debug("Comparing eigenvalue at position %i in result to %i in reference\n", i, comparedto);
            dd_debug("%e ?= %e\n", my_eigenvalues.data[i], known_eigenvalues[comparedto]);
            if (dd_doubles_close(my_eigenvalues.data[i], known_eigenvalues[comparedto], 1e-13)) {
                dd_debug("breaking out");
                found = true;
                break;
            }
        }
        if (!found) {
            FAILm("No match for eigenvalue returned by solver");
        }
        for (int j = 0; j < N; j++) {
            double result_value = my_eigenvectors.data[i * my_eigenvectors.rows + j];
            double reference_value = known_eigenvectors[comparedto * N + j];
            dd_debug("Comparing Col %i (%i) Row %i: %e (%e)\n", i, comparedto, j, result_value, reference_value);
            dd_debug("%e\n", result_value - reference_value);
            ASSERT_EQm("Eigenvector entry does not match", dd_doubles_close(result_value, reference_value, 1e-8), true);
        }
    }
    PASS();
}

TEST matrix_product(void) {
    int rows = 4;
    int cols = 3;
    double a[3*4] = {
        -1.16698056,  1.79731082,  1.36366154,  0.09013776,
        -0.88396565, -0.78706291,  0.9444493 ,  0.88359966,
         1.38745145,  0.52486817,  0.0954946 ,  0.90069779
    };
    dd_Matrix my_matrix = {
        .rows = rows,
        .cols = cols,
        .data = (double *)&a
    };
    double ref[4 * 4] = {
        4.06826042, -0.67346111, -2.29373313,  0.36341369,
       -0.67346111,  4.1252808 ,  1.7577047 , -0.06069535,
       -2.29373313,  1.7577047 ,  2.76067649,  1.04344425,
        0.36341369, -0.06069535,  1.04344425,  1.60012968
    };
    dd_Matrix ref_product = {
        .rows = rows,
        .cols = rows,
        .data = (double *)&ref
    };

    double output[4 * 4] = {
       0, 0, 0, 0,
       0, 0, 0, 0,
       0, 0, 0, 0,
       0, 0, 0, 0
    };
    dd_Matrix output_matrix = {
        .rows = rows,
        .cols = rows,
        .data = (double *)&output
    };

    dd_matrix_product(1.0, &my_matrix, false, &my_matrix, true, 0.0, &output_matrix);
    ASSERTm("Matrix product doesn't match reference", dd_doubles_all_close(output_matrix.data, ref_product.data, 4 * 4, 1e-8));
    PASS();
}

TEST row_mean(void) {
    int rows = 4;
    int cols = 3;
    double a[3 * 4] = {
        1.1, 0.9, 1, 1,
        2.1, 1.9, 2, 2,
        3.1, 2.9, 3, 3
    };
    dd_Matrix my_matrix = {
        .rows = rows,
        .cols = cols,
        .data = (double *)&a
    };
    double c[3 * 4] = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    dd_Matrix mean_sub_matrix = {
        .rows = rows,
        .cols = cols,
        .data = (double *)&c
    };
    double column_means[3] = {0, 0, 0};
    double right_means[3] = {1, 2, 3};
    double right_mean_sub[3 * 4] = {
        0.1, -0.1, 0, 0,
        0.1, -0.1, 0, 0,
        0.1, -0.1, 0, 0,
    };
    dd_subtract_mean_row(&my_matrix, &mean_sub_matrix, (double*)&column_means);
    ASSERTm("Column mean values incorrect", dd_doubles_all_close((double*)&column_means, (double*)&right_means, 3, 1e-8));
    ASSERTm("Mean subtracted matrix incorrect", dd_doubles_all_close(mean_sub_matrix.data, (double*)&right_mean_sub, 3 * 4, 1e-8));
    PASS();
}


TEST column_mean(void) {
    int rows = 4;
    int cols = 3;
    double a[3 * 4] = {
        1.1, 2.1, 3.1, 4.1,
        0.9, 1.9, 2.9, 3.9,
        1.0, 2, 3, 4
    };
    dd_Matrix my_matrix = {
        .rows = rows,
        .cols = cols,
        .data = (double *)&a
    };
    double c[3 * 4] = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    dd_Matrix mean_sub_matrix = {
        .rows = rows,
        .cols = cols,
        .data = (double *)&c
    };
    double row_means[3] = {0, 0, 0};
    double right_means[3] = {1, 2, 3};
    double right_mean_sub[3 * 4] = {
        0.1, 0.1, 0.1, 0.1,
        -0.1, -0.1, -0.1, -0.1,
        0.0, 0.0, 0.0, 0.0,
    };
    dd_subtract_mean_column(&my_matrix, &mean_sub_matrix, (double*)&row_means);
    ASSERTm("Row mean values incorrect", dd_doubles_all_close((double*)&row_means, (double*)&right_means, 3, 1e-8));
    ASSERTm("Mean subtracted matrix incorrect", dd_doubles_all_close(mean_sub_matrix.data, (double*)&right_mean_sub, 3 * 4, 1e-8));
    PASS();
}

TEST sample_covariance(void) {
    int rows = 4;
    int cols = 3;
    double a[3*4] = {
        -1.16698056,  1.79731082,  1.36366154,  0.09013776,
        -0.88396565, -0.78706291,  0.9444493 ,  0.88359966,
         1.38745145,  0.52486817,  0.0954946 ,  0.90069779
    };
    dd_Matrix my_matrix = {
        .rows = rows,
        .cols = cols,
        .data = (double *)&a
    };
    double c[4 * 4] = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };
    dd_Matrix cov_matrix = {
        .rows = rows,
        .cols = rows,
        .data = (double *)&c
    };
    dd_sample_covariance(&my_matrix, &cov_matrix);

    double ref[4 * 4] = {
         1.96075933, -0.16697364, -0.88106996,  0.3889865 ,
        -0.16697364,  1.66987684,  0.26388346, -0.50992695,
        -0.88106996,  0.26388346,  0.41745173, -0.22917832,
         0.3889865 , -0.50992695, -0.22917832,  0.21448028
    };
    // dd_print_matrix(&cov_matrix);
    ASSERTm("Sample covariance doesn't match reference", dd_doubles_all_close(cov_matrix.data, (double*)&ref, 4 * 4, 1e-8));
    PASS();
}

SUITE(suite)
{
    RUN_TEST(load_fits_cube);
    RUN_TEST(load_fits_image_float);
    RUN_TEST(load_fits_image_int_conversion);
    RUN_TEST(roundtrip_fits_cube);
    RUN_TEST(make_transpose);
    RUN_TEST(get_eigenvectors_mkl_syevr);
    RUN_TEST(get_eigenvector_range_mkl_syevr);
    RUN_TEST(matrix_product);
    RUN_TEST(column_mean);
    RUN_TEST(row_mean);
    RUN_TEST(sample_covariance);
}

/* Add definitions that need to be in the test runner's main file. */
GREATEST_MAIN_DEFS();

/* Set up, run suite(s) of tests, report pass/fail/skip stats. */
int run_tests(void)
{
    GREATEST_INIT(); /* init. greatest internals */
    /* List of suites to run (if any). */
    RUN_SUITE(suite);

    GREATEST_PRINT_REPORT(); /* display results */
    return greatest_all_passed();
}

/* main(), for a standalone command-line test runner.
 * This replaces run_tests above, and adds command line option
 * handling and exiting with a pass/fail status. */
int main(int argc, char **argv)
{
    GREATEST_MAIN_BEGIN(); /* init & parse command-line args */
    RUN_SUITE(suite);
    GREATEST_MAIN_END(); /* display results */
}