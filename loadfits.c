#include <stdio.h>
#include <getopt.h>
#include "fitsio.h"

const int ANY_EXTVER = 0;

void check_status(int status)
{
    if (status)
    {
        fits_report_error(stderr, status);
        exit(status);
    }
}

int main(int argc, char *argv[])
{
    fitsfile *infptr, *outfptr; /* FITS file pointers defined in fitsio.h */
    int status = 0;             /* status must always be initialized = 0  */
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s INPUT_FILE OUTPUT_FILE\n", argv[0]);
        exit(1);
    }
    char *input_file = argv[1];
    char *output_file = argv[2];
    fits_open_image(&infptr, input_file, READONLY, &status);
    check_status(status);

    int naxis;
    fits_get_img_dim(infptr, &naxis, &status);
    check_status(status);
    printf("%i dimensions\n", naxis);

    long imageDims[naxis];
    fits_get_img_size(infptr, naxis, (long *)&imageDims, &status);
    check_status(status);
    int total_n_elements = imageDims[0] * imageDims[1];
    printf("Image is %li ", imageDims[naxis - 1]);
    for (int i = naxis - 2; i >= 0; i--)
    {
        printf("x %li ", imageDims[i]);
    }
    printf("\n");

    double *imageMem = (double *)malloc(total_n_elements * sizeof(double));
    double **image = (double **)malloc(imageDims[1] * sizeof(size_t));
    if (image == NULL || imageMem == NULL)
    {
        fprintf(stderr, "Unable to allocate RAM\n");
        exit(1);
    }
    for (size_t i = 0; i < imageDims[1]; i++)
    {
        image[i] = &imageMem[imageDims[0] * i];
    }

    long first_pixel_indices[2] = {1, 1};
    fits_read_pix(infptr, TDOUBLE, first_pixel_indices, total_n_elements,
                  0, imageMem, 0, &status);
    check_status(status);
    for (int i = 0; i < imageDims[1]; i++)
    {
        for (int j = 0; j < imageDims[0]; j++)
        {
            printf("%f ", image[i][j]);
        }
        printf("\n");
    }

    fits_create_diskfile(&outfptr, output_file, &status); // use _diskfile to avoid 'smart' filename parsing
    check_status(status);

    fits_create_img(outfptr, LONGLONG_IMG, naxis, imageDims, &status);
    check_status(status);

    fits_write_pix(outfptr, TDOUBLE, first_pixel_indices, total_n_elements, imageMem, &status);
    check_status(status);

    fits_close_file(outfptr, &status);
    fits_close_file(infptr, &status);
    return 0;
}