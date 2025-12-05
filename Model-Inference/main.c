#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "rknn_api.h"

#define INPUT_LENGTH 2
#define TEMP_MIN 26.7
#define HUM_MIN 40.01
#define TEMP_RANGE 13.3
#define HUM_RANGE 59.97

#define OUTPUT_MAX 105.72
#define OUTPUT_MIN 26.79
#define OUTPUT_RANGE 78.93

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <rknn_model_path>\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    rknn_context ctx;
    int ret;
    float static_input[INPUT_LENGTH];
    
    //Static input: temperature=25.0, humidity=50.0
    static_input[0] = (29.0 - TEMP_MIN) / TEMP_RANGE;
    static_input[1] = (50.0 - HUM_MIN) / HUM_RANGE;

    // 1. Initialize RKNN context
    ret = rknn_init(&ctx, (char*)model_path, 0, 0, NULL);
    if (ret < 0) {
        printf("rknn_init failed! ret=%d\n", ret);
        return -1;
    }

    // 2. Query input/output number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query RKNN_QUERY_IN_OUT_NUM failed! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }

    printf("Model inputs=%d, outputs=%d\n", io_num.n_input, io_num.n_output);

    // 3. Prepare input tensor
    rknn_tensor_attr input_attrs[io_num.n_input];
    rknn_tensor_mem* input_mems[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query input attr %d failed! ret=%d\n", i, ret);
            rknn_destroy(ctx);
            return -1;
        }

        printf("Input %d: name=%s, size=%d\n", i, input_attrs[i].name, input_attrs[i].size_with_stride);

        // Allocate memory (pass-through handled by RKNN runtime for INT8)
        input_mems[i] = rknn_create_mem(ctx, input_attrs[i].size_with_stride);
        if (!input_mems[i]) {
            printf("rknn_create_mem input %d failed!\n", i);
            rknn_destroy(ctx);
            return -1;
        }

        // Fill input with quantized data
        int8_t* in_data = (int8_t*)input_mems[i]->virt_addr;
        for (int j = 0; j < input_attrs[i].n_elems; j++) {
            in_data[j] = (int8_t)((static_input[j]/input_attrs[i].scale) + input_attrs[i].zp);
        }

        printf("Input data temp = %d  hum %d\n",in_data[0],in_data[1]);

        ret = rknn_set_io_mem(ctx, input_mems[i], &input_attrs[i]);
        if (ret < 0) {
            printf("rknn_set_io_mem input %d failed! ret=%d\n", i, ret);
            rknn_destroy(ctx);
            return -1;
        }
    }

    // 4. Prepare output tensor
    rknn_tensor_attr output_attrs[io_num.n_output];
    rknn_tensor_mem* output_mems[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query output attr %d failed! ret=%d\n", i, ret);
            rknn_destroy(ctx);
            return -1;
        }

        printf("Output %d: name=%s, size=%d\n", i, output_attrs[i].name, output_attrs[i].size_with_stride);

        // Allocate memory
        output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        if (!output_mems[i]) {
            printf("rknn_create_mem output %d failed!\n", i);
            rknn_destroy(ctx);
            return -1;
        }

        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("rknn_set_io_mem output %d failed! ret=%d\n", i, ret);
            rknn_destroy(ctx);
            return -1;
        }
    }

    // 5. Run inference
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        printf("rknn_run failed! ret=%d\n", ret);
    } else {
        printf("Inference run successfully!\n");
    }


    printf("\nScale=%f zp=%d n_elems=%d\n",
        output_attrs[0].scale, output_attrs[0].zp, output_attrs[0].n_elems);

    // 6. Rescale INT8 output back to FP32 and save to file
    if (io_num.n_output > 0) {
        int8_t* out_data_int8 = (int8_t*)output_mems[0]->virt_addr;
        float* out_data_fp32 = (float*)malloc(output_attrs[0].n_elems * sizeof(float));

        if (!out_data_fp32) {
            printf("Failed to allocate memory for FP32 output\n");
            rknn_destroy(ctx);
            return -1;
        }
        for (int i = 0; i < output_attrs[0].n_elems; i++) {
            int8_t q = out_data_int8[i];
            printf("Output Data h_index %d\n",q);
            out_data_fp32[i] = ((q - output_attrs[0].zp) * output_attrs[0].scale);
        }
    
        out_data_fp32[0] = out_data_fp32[0] * OUTPUT_RANGE + OUTPUT_MIN;
        // Save to text file as a single line
        FILE* f = fopen("output.txt", "w");
        if (!f) {
            printf("Failed to open output.txt\n");
            free(out_data_fp32);
            rknn_destroy(ctx);
            return -1;
        }

        for (int i = 0; i < output_attrs[0].n_elems; i++) {
            fprintf(f, "%.6f ", out_data_fp32[i]);
        }
        fclose(f);
        printf("Rescaled FP32 output saved to output.txt\n");

        free(out_data_fp32);
    }
    printf("\n------ OUTPUT ATTRIBUTES ------\n");
    printf("type        = %d\n", output_attrs[0].type);
    printf("fmt         = %d\n", output_attrs[0].fmt);
    printf("n_elems     = %d\n", output_attrs[0].n_elems);
    printf("scale       = %f\n", output_attrs[0].scale);
    printf("zero point  = %d\n", output_attrs[0].zp);
    printf("--------------------------------\n");

    printf("\n------ INPUT ATTRIBUTES ------\n");
    printf("type        = %d\n", input_attrs[0].type);
    printf("fmt         = %d\n", input_attrs[0].fmt);
    printf("n_elems     = %d\n", input_attrs[0].n_elems);
    printf("scale       = %f\n", input_attrs[0].scale);
    printf("zero point  = %d\n", input_attrs[0].zp);
    printf("--------------------------------\n");


    // 7. Release resources
    for (int i = 0; i < io_num.n_input; i++) {
        rknn_destroy_mem(ctx, input_mems[i]);
        free(input_mems[i]);
    }
    for (int i = 0; i < io_num.n_output; i++) {
        rknn_destroy_mem(ctx, output_mems[i]);
        free(output_mems[i]);
    }
    rknn_destroy(ctx);

    return 0;
}
