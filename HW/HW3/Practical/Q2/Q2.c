#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bmp.c"
#include <assert.h>
#include <immintrin.h>

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

typedef unsigned int int32;
typedef short int16;
typedef unsigned char byte;

int main()
{
    /* start reading the file and its information*/
    byte *pixels_top, *pixels_bg;
    int32 width_top, width_bg;
    int32 height_top, height_bg;
    int32 bytesPerPixel_top, bytesPerPixel_bg;
    ReadImage("dino.bmp", &pixels_top, &width_top, &height_top, &bytesPerPixel_top);
    ReadImage("parking.bmp", &pixels_bg, &width_bg, &height_bg, &bytesPerPixel_bg);

    /* images should have color and be of the same size */
    assert(bytesPerPixel_top == 3);
    assert(width_top == width_bg);
    assert(height_top == height_bg);
    assert(bytesPerPixel_top == bytesPerPixel_bg);

    /* we can now work with one size */
    int32 width = width_top, height = height_top, bytesPerPixel = bytesPerPixel_top;

    __m256i _green_dist, _red_dist, _blue_dist, _temp1, _temp2, _one, _bg, _cmp;
    __m256i _blend_red, _blend_green, _blend_blue;
    __m128i _blend_red_small, _blend_green_small, _blend_blue_small;
    _one = _mm256_set1_epi16(255);

    /* These are used to seperate red, green, and blue pixels from RGB-vectors */
    __m128i seperate_red_idx_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0);  // RGBRGBRGBRGBRGBR to ----------RRRRRR
    __m128i seperate_red_idx_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1); // GBRGBRGBRGBRGBRG to -----RRRRR------
    __m128i seperate_red_idx_2 = _mm_set_epi8(13, 10, 7, 4, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1); // BRGBRGBRGBRGBRGB to RRRRR----------
    __m128i seperate_green_idx_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1);
    __m128i seperate_green_idx_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1);
    __m128i seperate_green_idx_2 = _mm_set_epi8(14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i seperate_blue_idx_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2);
    __m128i seperate_blue_idx_1 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1, -1, -1, -1, -1, -1);
    __m128i seperate_blue_idx_2 = _mm_set_epi8(15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    /* These are used to merge red, green, and blue pixels into RGB-vectors */
    __m128i merge_red_idx_0 = _mm_set_epi8(5, -1, -1, 4, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0);       // RRRRRRRRRRRRRRRR to R--R--R--R--R--R
    __m128i merge_red_idx_1 = _mm_set_epi8(-1, 10, -1, -1, 9, -1, -1, 8, -1, -1, 7, -1, -1, 6, -1, -1);     // RRRRRRRRRRRRRRRR to -R--R--R--R--R--
    __m128i merge_red_idx_2 = _mm_set_epi8(-1, -1, 15, -1, -1, 14, -1, -1, 13, -1, -1, 12, -1, -1, 11, -1); // RRRRRRRRRRRRRRRR to --R--R--R--R--R-
    __m128i merge_green_idx_0 = _mm_set_epi8(-1, -1, 4, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0, -1);
    __m128i merge_green_idx_1 = _mm_set_epi8(10, -1, -1, 9, -1, -1, 8, -1, -1, 7, -1, -1, 6, -1, -1, 5);
    __m128i merge_green_idx_2 = _mm_set_epi8(-1, 15, -1, -1, 14, -1, -1, 13, -1, -1, 12, -1, -1, 11, -1, -1);
    __m128i merge_blue_idx_0 = _mm_set_epi8(-1, 4, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0, -1, -1);
    __m128i merge_blue_idx_1 = _mm_set_epi8(-1, -1, 9, -1, -1, 8, -1, -1, 7, -1, -1, 6, -1, -1, 5, -1);
    __m128i merge_blue_idx_2 = _mm_set_epi8(15, -1, -1, 14, -1, -1, 13, -1, -1, 12, -1, -1, 11, -1, -1, 10);

    /* start replacing green_top screen */
    for (int start = 0; start < width * height; start += 16)
    {
        int idx = start * bytesPerPixel;

        /* load top image pixels */
        const __m128i chunk0_top = _mm_loadu_si128((const __m128i *)(pixels_top + idx));      //  |RGB|RGB|RGB|RGB|RGB|R
        const __m128i chunk1_top = _mm_loadu_si128((const __m128i *)(pixels_top + idx + 16)); //  GB|RGB|RGB|RGB|RGB|RG
        const __m128i chunk2_top = _mm_loadu_si128((const __m128i *)(pixels_top + idx + 32)); //  B|RGB|RGB|RGB|RGB|RGB|

        /* Create red, green, and blue vectors for top image and extend them to 16 bits. */
        const __m256i red_top = _mm256_cvtepu8_epi16( //  0R 0R 0R 0R 0R 0R 0R 0R 0R 0R 0R 0R 0R 0R 0R 0R
            _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(chunk0_top, seperate_red_idx_0),  //  ----------RRRRRR
                    _mm_shuffle_epi8(chunk1_top, seperate_red_idx_1)), //  -----RRRRR------
                _mm_shuffle_epi8(chunk2_top, seperate_red_idx_2)));    //  RRRRR----------

        const __m256i blue_top = _mm256_cvtepu8_epi16( //  0B 0B 0B 0B 0B 0B 0B 0B 0B 0B 0B 0B 0B 0B 0B 0B
            _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(chunk0_top, seperate_blue_idx_0),  //  -----------BBBBB
                    _mm_shuffle_epi8(chunk1_top, seperate_blue_idx_1)), //  -----BBBBBB-----
                _mm_shuffle_epi8(chunk2_top, seperate_blue_idx_2)));    //  BBBBB-----------

        const __m256i green_top = _mm256_cvtepu8_epi16( //  0G 0G 0G 0G 0G 0G 0G 0G 0G 0G 0G 0G 0G 0G 0G 0G
            _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(chunk0_top, seperate_green_idx_0),  //  -----------GGGGG
                    _mm_shuffle_epi8(chunk1_top, seperate_green_idx_1)), //  ------GGGGG-----
                _mm_shuffle_epi8(chunk2_top, seperate_green_idx_2)));    //  GGGGGG----------

        /* load background image pixels */
        const __m128i chunk0_bg = _mm_loadu_si128((const __m128i *)(pixels_bg + idx));
        const __m128i chunk1_bg = _mm_loadu_si128((const __m128i *)(pixels_bg + idx + 16));
        const __m128i chunk2_bg = _mm_loadu_si128((const __m128i *)(pixels_bg + idx + 32));

        /* Create red, green, and blue vectors for background image and extend them to 16 bits. */
        const __m256i red_bg = _mm256_cvtepu8_epi16(
            _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(chunk0_bg, seperate_red_idx_0),
                    _mm_shuffle_epi8(chunk1_bg, seperate_red_idx_1)),
                _mm_shuffle_epi8(chunk2_bg, seperate_red_idx_2)));

        const __m256i blue_bg = _mm256_cvtepu8_epi16(
            _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(chunk0_bg, seperate_blue_idx_0),
                    _mm_shuffle_epi8(chunk1_bg, seperate_blue_idx_1)),
                _mm_shuffle_epi8(chunk2_bg, seperate_blue_idx_2)));

        const __m256i green_bg = _mm256_cvtepu8_epi16(
            _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(chunk0_bg, seperate_green_idx_0),
                    _mm_shuffle_epi8(chunk1_bg, seperate_green_idx_1)),
                _mm_shuffle_epi8(chunk2_bg, seperate_green_idx_2)));

        /* Compare RGB in each pixel to decide whether it is green or not */
        _green_dist = _mm256_adds_epi16(_one, _mm256_adds_epi16(red_top, blue_top));
        _blue_dist = _mm256_adds_epi16(_one, _mm256_adds_epi16(red_top, green_top));
        _red_dist = _mm256_adds_epi16(_one, _mm256_adds_epi16(green_top, blue_top));

        _cmp = _mm256_and_si256(_mm256_cmpgt_epi16(_red_dist, _green_dist), _mm256_cmpgt_epi16(_blue_dist, _green_dist));

        /* Merge foreground and background images based on compare result */
        _blend_red = _mm256_blendv_epi8(red_top, red_bg, _cmp);
        _blend_blue = _mm256_blendv_epi8(blue_top, blue_bg, _cmp);
        _blend_green = _mm256_blendv_epi8(green_top, green_bg, _cmp);

        /* convert 256 bit vector to 128 */
        _blend_red_small = _mm_packus_epi16(_mm256_extractf128_si256(_blend_red, 0), _mm256_extractf128_si256(_blend_red, 1));
        _blend_blue_small = _mm_packus_epi16(_mm256_extractf128_si256(_blend_blue, 0), _mm256_extractf128_si256(_blend_blue, 1));
        _blend_green_small = _mm_packus_epi16(_mm256_extractf128_si256(_blend_green, 0), _mm256_extractf128_si256(_blend_green, 1));

        /* Merge red, green, and blue vectors to get RGB-vector */
        const __m128i new_chunk0 = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(_blend_red_small, merge_red_idx_0),
                _mm_shuffle_epi8(_blend_blue_small, merge_blue_idx_0)),
            _mm_shuffle_epi8(_blend_green_small, merge_green_idx_0));

        const __m128i new_chunk1 = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(_blend_red_small, merge_red_idx_1),
                _mm_shuffle_epi8(_blend_blue_small, merge_blue_idx_1)),
            _mm_shuffle_epi8(_blend_green_small, merge_green_idx_1));

        const __m128i new_chunk2 = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(_blend_red_small, merge_red_idx_2),
                _mm_shuffle_epi8(_blend_blue_small, merge_blue_idx_2)),
            _mm_shuffle_epi8(_blend_green_small, merge_green_idx_2));

        /* Store result */
        _mm_storeu_si128((const __m128i *)(pixels_top + idx), new_chunk0);
        _mm_storeu_si128((const __m128i *)(pixels_top + idx + 16), new_chunk1);
        _mm_storeu_si128((const __m128i *)(pixels_top + idx + 32), new_chunk2);
    }

    /* write new image */
    WriteImage("replaced.bmp", pixels_top, width, height, bytesPerPixel);

    int kernel_size = 5;
    float a = -1.0 / 8;
    float kernel[5][5] = {{0, 0, 0, 0, 0},
                          {0, a, a, a, 0},
                          {0, a, 2.0, a, 0},
                          {0, a, a, a, 0},
                          {0, 0, 0, 0, 0}};

    printf("kernel %f\n", kernel[1][2]);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int start = i * width + j;
            // printf("ij %d %d\n",i , j);
            for (int rgb = 0; rgb < 3; rgb++)
            {
                float sum = 0;
                // printf("--------\n");
                for (int k = -kernel_size / 2; k <= kernel_size / 2; k++)
                    for (int p = -kernel_size / 2; p <= kernel_size / 2; p++)
                    {
                        if (i + k < 0 || i + k >= height || p + j < 0 || p + j > width)
                            continue;

                        int idx = (i + k) * width + (j + p);
                        // printf("compare %f %f\n",kernel[k + kernel_size / 2][p + kernel_size / 2], (float)pixels_top[idx * bytesPerPixel + rgb] );
                        sum += kernel[k + kernel_size / 2][p + kernel_size / 2] * pixels_top[idx * bytesPerPixel + rgb];
                    }
                if (sum > 255)
                    sum = 255;
                if (sum < 0)
                    sum = 0;

                // printf(">>>> %d %d\n", (byte)pixels_top[start * bytesPerPixel + rgb], (byte)sum);
                pixels_top[start * bytesPerPixel + rgb] = sum;
            }
            // pixels_top[start * bytesPerPixel] = pixels_bg[start * bytesPerPixel];
            // pixels_top[start * bytesPerPixel + 1] = pixels_bg[start * bytesPerPixel + 1];
            // pixels_top[start * bytesPerPixel + 2] = pixels_bg[start * bytesPerPixel + 2];
        }
    }

    /* write new image */
    WriteImage("sharpened.bmp", pixels_top, width, height, bytesPerPixel);

    /* free everything */
    free(pixels_top);
    free(pixels_bg);
    return 0;
}
