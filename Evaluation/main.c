#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "rknn_api.h"

#define NUM_SAMPLES 1000

#define INPUT_LENGTH 2

#define TEMP_MIN 26.7
#define HUM_MIN 40.01
#define TEMP_RANGE 13.3
#define HUM_RANGE 59.97

#define OUTPUT_MAX 105.72
#define OUTPUT_MIN 26.79
#define OUTPUT_RANGE 78.93

static inline double time_diff_us(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1000000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000.0;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s <rknn_model_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    rknn_context ctx;
    int ret;

    float samples[NUM_SAMPLES][2];

    FILE *fin = fopen("input.txt", "r");
    if (!fin)
    {
        printf("Failed to open input.txt\n");
        return -1;
    }

    int count = 0;
    while (count < NUM_SAMPLES &&
           fscanf(fin, "%f %f", &samples[count][0], &samples[count][1]) == 2)
    {
        count++;
    }
    fclose(fin);

    if (count == 0)
    {
        printf("No samples found in input.txt\n");
        return -1;
    }

    printf("Loaded %d samples\n", count);

    FILE *fout = fopen("output.txt", "w");
    if (!fout)
    {
        printf("Failed to open output.txt\n");
        return -1;
    }

    fprintf(fout, "Temp,Hum,Output,Time_us\n");

    // 1. Initialize RKNN context
    ret = rknn_init(&ctx, (char *)model_path, 0, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init failed! ret=%d\n", ret);
        return -1;
    }

    // 2. Query input/output number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query RKNN_QUERY_IN_OUT_NUM failed! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }

    // 3. Prepare input tensor
    rknn_tensor_attr input_attrs;
    rknn_tensor_mem *input_mems;

    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs, sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query input attr failed! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }

    // Allocate memory (pass-through handled by RKNN runtime for INT8)
    input_mems = rknn_create_mem(ctx, input_attrs.size);
    if (!input_mems)
    {
        printf("rknn_create_mem input failed!\n");
        rknn_destroy(ctx);
        return -1;
    }

    // Fill input with quantized data
    int8_t *in_data = (int8_t *)input_mems->virt_addr;

    // 4. Prepare output tensor
    rknn_tensor_attr output_attrs;
    rknn_tensor_mem *output_mems;

    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs, sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query output attr failed! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }

    output_mems = rknn_create_mem(ctx, output_attrs.size);
    if (!output_mems)
    {
        printf("rknn_create_mem output failed!\n");
        rknn_destroy(ctx);
        return -1;
    }

    ret = rknn_set_io_mem(ctx, output_mems, &output_attrs);
    if (ret < 0)
    {
        printf("rknn_set_io_mem output failed! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }
    ret = rknn_set_io_mem(ctx, input_mems, &input_attrs);
    if (ret < 0)
    {
        printf("rknn_set_io_mem input failed! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }

    // Warmup (important for fair timing)
    for (int i = 0; i < 5; i++)
        rknn_run(ctx, NULL);

    // Batch inference
    for (int i = 0; i < count; i++)
    {
        float temp = samples[i][0];
        float hum = samples[i][1];

        // Normalize
        float norm_temp = (temp - TEMP_MIN) / TEMP_RANGE;
        float norm_hum = (hum - HUM_MIN) / HUM_RANGE;

        // Quantize to int8
        int8_t *in_data = (int8_t *)input_mems->virt_addr;
        in_data[0] = (int8_t)((norm_temp / input_attrs.scale) + input_attrs.zp);
        in_data[1] = (int8_t)((norm_hum / input_attrs.scale) + input_attrs.zp);


        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        ret = rknn_run(ctx, NULL);

        clock_gettime(CLOCK_MONOTONIC, &end);

        if (ret < 0)
        {
            printf("rknn_run failed at index %d\n", i);
            continue;
        }

        double time_us = time_diff_us(start, end);

        // Get output
        int8_t *q = (int8_t *)output_mems->virt_addr;
        float out_norm = (q[0] - output_attrs.zp) * output_attrs.scale;

        // Denormalize
        float out_real = out_norm * OUTPUT_RANGE + OUTPUT_MIN;

        // Save to file
        fprintf(fout, "%.2f,%.2f,%.5f,%.2f\n",
                temp, hum, out_real, time_us);

        if (i % 100 == 0)
            printf("Processed %d / %d\n", i, count);
    }

    fclose(fout);

    // 7. Release resources
    rknn_destroy_mem(ctx, input_mems);
    rknn_destroy_mem(ctx, output_mems);
    rknn_destroy(ctx);

    return 0;
}
