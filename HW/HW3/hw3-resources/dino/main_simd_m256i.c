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

int is_green(byte r, byte g, byte b)
{
    int green_dist = (byte)255 + r + b,
        red_dist = (byte)255 + g + b,
        blue_dist = (byte)255 + r + g;

    printf("IS GREEN rgb: %d %d %d, dist: %d %d %d\n", r, g, b, red_dist, green_dist, blue_dist);
    return (green_dist < red_dist && green_dist < blue_dist);
}

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

    __m256i _green_dist, _red_dist, _blue_dist, _cmp, _blend_red, _blend_green, _blend_blue;

    /* These are used to seperate red, green, and blue pixels from RGB-vectors */
    const __m256i seperate_red_idx_0 = _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // e.g. RGBRG to ---RR
                                                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                       30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0);
    const __m256i seperate_red_idx_1 = _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // e.g. BRGBR to -RR--
                                                       29, 26, 23, 20, 17, 14, 11, 8, 5, 2,
                                                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i seperate_red_idx_2 = _mm256_set_epi8(31, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1, // e.g. GBRGB to R----
                                                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i seperate_green_idx_0 = _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                         31, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1);
    const __m256i seperate_green_idx_1 = _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                         30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0,
                                                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i seperate_green_idx_2 = _mm256_set_epi8(29, 26, 23, 20, 17, 14, 11, 8, 5, 2,
                                                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i seperate_blue_idx_0 = _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                        29, 26, 23, 20, 17, 14, 11, 8, 5, 2);
    const __m256i seperate_blue_idx_1 = _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                        31, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1,
                                                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i seperate_blue_idx_2 = _mm256_set_epi8(30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0,
                                                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    /* These are used to merge red, green, and blue pixels into RGB-vectors */
    const __m256i merge_red_idx_0 = _mm256_set_epi8(-1, 10, -1, -1, 9, -1, -1, 8, -1, -1, 7, -1, -1, 6, -1, -1, 5, -1, -1, 4, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0);           // RRRRRR...RRR to R--R--...R--R--R
    const __m256i merge_red_idx_1 = _mm256_set_epi8(21, -1, -1, 20, -1, -1, 19, -1, -1, 18, -1, -1, 17, -1, -1, 16, -1, -1, 15, -1, -1, 14, -1, -1, 13, -1, -1, 12, -1, -1, 11, -1); // RRRRRR...RRR to -R--R-...R--R--
    const __m256i merge_red_idx_2 = _mm256_set_epi8(-1, -1, 31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1, -1, 27, -1, -1, 26, -1, -1, 25, -1, -1, 24, -1, -1, 23, -1, -1, 22, -1, -1); // RRRRRR...RRR to --R--R--...R--R-
    const __m256i merge_green_idx_0 = _mm256_set_epi8(10, -1, -1, 9, -1, -1, 8, -1, -1, 7, -1, -1, 6, -1, -1, 5, -1, -1, 4, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0, -1);
    const __m256i merge_green_idx_1 = _mm256_set_epi8(-1, -1, 20, -1, -1, 19, -1, -1, 18, -1, -1, 17, -1, -1, 16, -1, -1, 15, -1, -1, 14, -1, -1, 13, -1, -1, 12, -1, -1, 11, -1, -1);
    const __m256i merge_green_idx_2 = _mm256_set_epi8(-1, 31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1, -1, 27, -1, -1, 26, -1, -1, 25, -1, -1, 24, -1, -1, 23, -1, -1, 22, -1, -1, 21);
    const __m256i merge_blue_idx_0 = _mm256_set_epi8(-1, -1, 9, -1, -1, 8, -1, -1, 7, -1, -1, 6, -1, -1, 5, -1, -1, 4, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0, -1, -1);
    const __m256i merge_blue_idx_1 = _mm256_set_epi8(-1, 20, -1, -1, 19, -1, -1, 18, -1, -1, 17, -1, -1, 16, -1, -1, 15, -1, -1, 14, -1, -1, 13, -1, -1, 12, -1, -1, 11, -1, -1, 10);
    const __m256i merge_blue_idx_2 = _mm256_set_epi8(31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1, -1, 27, -1, -1, 26, -1, -1, 25, -1, -1, 24, -1, -1, 23, -1, -1, 22, -1, -1, 21, -1);

    /* start replacing green_top screen */
    for (int start = 0; start < width * height; start += 32)
    {
        int idx = start * bytesPerPixel;

        /* load top image pixels */
        const __m256i chunk0_top = _mm256_loadu_si256((__m256i *)(pixels_top + idx));      //  |RGB|RGB|RGB|RGB|RGB|RGB|RGB|RGB|RGB|RGB|RG
        const __m256i chunk1_top = _mm256_loadu_si256((__m256i *)(pixels_top + idx + 32)); //  B|RGB|RGB|RGB|RGB|RGB|RGB|RGB|RGB|RGB|RGB|R
        const __m256i chunk2_top = _mm256_loadu_si256((__m256i *)(pixels_top + idx + 64)); //  GB|RGB|RGB|RGB|RGB|RGB|RGB|RGB|RGB|RGB|RGB|

        /* Create red, green, and blue vectors for top image and extend them to 16 bits. */
        const __m256i red_top =
            _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_shuffle_epi8(chunk0_top, seperate_red_idx_0),  //  ----------RRRRRR
                    _mm256_shuffle_epi8(chunk1_top, seperate_red_idx_1)), //  -----RRRRR------
                _mm256_shuffle_epi8(chunk2_top, seperate_red_idx_2));     //  RRRRR----------

        const __m256i blue_top =
            _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_shuffle_epi8(chunk0_top, seperate_blue_idx_0),  //  -----------BBBBB
                    _mm256_shuffle_epi8(chunk1_top, seperate_blue_idx_1)), //  -----BBBBBB-----
                _mm256_shuffle_epi8(chunk2_top, seperate_blue_idx_2));     //  BBBBB-----------

        const __m256i green_top =
            _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_shuffle_epi8(chunk0_top, seperate_green_idx_0),  //  -----------GGGGG
                    _mm256_shuffle_epi8(chunk1_top, seperate_green_idx_1)), //  ------GGGGG-----
                _mm256_shuffle_epi8(chunk2_top, seperate_green_idx_2));     //  GGGGGG----------

        /* load background image pixels */
        const __m256i chunk0_bg = _mm256_loadu_si256((__m256i *)(pixels_bg + idx));
        const __m256i chunk1_bg = _mm256_loadu_si256((__m256i *)(pixels_bg + idx + 32));
        const __m256i chunk2_bg = _mm256_loadu_si256((__m256i *)(pixels_bg + idx + 64));

        /* Create red, green, and blue vectors for background image and extend them to 16 bits. */
        const __m256i red_bg =
            _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_shuffle_epi8(chunk0_bg, seperate_red_idx_0),
                    _mm256_shuffle_epi8(chunk1_bg, seperate_red_idx_1)),
                _mm256_shuffle_epi8(chunk2_bg, seperate_red_idx_2));

        const __m256i blue_bg =
            _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_shuffle_epi8(chunk0_bg, seperate_blue_idx_0),
                    _mm256_shuffle_epi8(chunk1_bg, seperate_blue_idx_1)),
                _mm256_shuffle_epi8(chunk2_bg, seperate_blue_idx_2));

        const __m256i green_bg =
            _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_shuffle_epi8(chunk0_bg, seperate_green_idx_0),
                    _mm256_shuffle_epi8(chunk1_bg, seperate_green_idx_1)),
                _mm256_shuffle_epi8(chunk2_bg, seperate_green_idx_2));

        byte *res = (byte *)&red_bg;
        printf("red_bg: %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x\n",
               res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], res[12], res[13], res[14], res[15],
               res[16], res[17], res[18], res[19], res[20], res[21], res[22], res[23], res[24], res[25], res[26], res[27], res[28], res[29], res[30], res[31]);

        res = (byte *)&red_top;
        printf("red_top: %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x\n",
               res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], res[12], res[13], res[14], res[15],
               res[16], res[17], res[18], res[19], res[20], res[21], res[22], res[23], res[24], res[25], res[26], res[27], res[28], res[29], res[30], res[31]);
        /* Compare RGB in each pixel to decide whether it is green or not */
        _green_dist = _mm256_adds_epi8(red_top, blue_top);
        _blue_dist = _mm256_adds_epi8(red_top, green_top);
        _red_dist = _mm256_adds_epi8(green_top, blue_top);

        // TODO:  this needs _mm256_cmpgt_epu8_mask
        _cmp = _mm256_and_si256(_mm256_cmpgt_epi8(_red_dist, _green_dist), _mm256_cmpgt_epi8(_blue_dist, _green_dist));
        res = (byte *)&_cmp;
        printf("_cmp: %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x\n",
               res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], res[12], res[13], res[14], res[15],
               res[16], res[17], res[18], res[19], res[20], res[21], res[22], res[23], res[24], res[25], res[26], res[27], res[28], res[29], res[30], res[31]);
        /* Merge foreground and background images based on compare result */
        _blend_red = _mm256_blendv_epi8(red_top, red_bg, _cmp);
        _blend_blue = _mm256_blendv_epi8(blue_top, blue_bg, _cmp);
        _blend_green = _mm256_blendv_epi8(green_top, green_bg, _cmp);

        res = (byte *)&_blend_red;
        printf("_blend_red: %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x %0x\n",
               res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], res[12], res[13], res[14], res[15],
               res[16], res[17], res[18], res[19], res[20], res[21], res[22], res[23], res[24], res[25], res[26], res[27], res[28], res[29], res[30], res[31]);

        /* Merge red, green, and blue vectors to get RGB-vector */
        const __m256i new_chunk0 = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(_blend_red, merge_red_idx_0),
                _mm256_shuffle_epi8(_blend_blue, merge_blue_idx_0)),
            _mm256_shuffle_epi8(_blend_green, merge_green_idx_0));

        const __m256i new_chunk1 = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(_blend_red, merge_red_idx_1),
                _mm256_shuffle_epi8(_blend_blue, merge_blue_idx_1)),
            _mm256_shuffle_epi8(_blend_green, merge_green_idx_1));

        const __m256i new_chunk2 = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(_blend_red, merge_red_idx_2),
                _mm256_shuffle_epi8(_blend_blue, merge_blue_idx_2)),
            _mm256_shuffle_epi8(_blend_green, merge_green_idx_2));

        /* Store result */
        _mm256_storeu_si256((__m256i *)(pixels_top + idx), new_chunk0);
        _mm256_storeu_si256((__m256i *)(pixels_top + idx + 32), new_chunk1);
        _mm256_storeu_si256((__m256i *)(pixels_top + idx + 64), new_chunk2);
        // break;
    }

    /* write new image */
    WriteImage("replaced.bmp", pixels_top, width, height, bytesPerPixel);

    /* free everything */
    free(pixels_top);
    free(pixels_bg);
    return 0;
}
