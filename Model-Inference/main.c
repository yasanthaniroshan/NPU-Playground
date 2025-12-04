#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "rknn_api.h"

#define BATCH 1
#define CHANNEL 1
#define HEIGHT 32
#define WIDTH 32

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <rknn_model_path>\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    rknn_context ctx;
    int ret;
    float static_input[HEIGHT * WIDTH];
    for (int row = 0; row < HEIGHT; row++) {
        for (int col = 0; col < WIDTH; col++) {
            static_input[row * WIDTH + col] = (row == col) ? 255.0f : 0.0f;
        }
    }
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


        int number_of_negatives = 0;
        int number_of_zeros = 0;
        int number_of_positives = 0;

        // Fill input with neutral INT8 values
        int8_t* in_data = (int8_t*)input_mems[i]->virt_addr;
        for (int j = 0; j < input_attrs[i].n_elems; j++) {
            in_data[j] = (int8_t)(static_input[j] + input_attrs[i].zp);
            if(in_data[j] > 0) {
                number_of_positives++;
            }
            else if(in_data[j] <  0) {
                number_of_negatives++;
            }
            else {
                number_of_zeros++;
            }
        }

        printf("\n------ OUTPUT ATTRIBUTES ------\n");
        printf("number_of_negatives = %d\n",number_of_negatives);
        printf("number_of_zeros = %d\n",number_of_zeros);
        printf("number_of_positives = %d\n",number_of_positives );


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

    // print first 40 raw int8 values (stepping by 2)
    int8_t* out = (int8_t*)output_mems[0]->virt_addr;
    printf("First 40 raw int8 (NC1HWC2, take every 2nd byte):\n");
    for (int i = 0; i < 40; i += 2) {
        printf("%d ", out[i]);
    }
    printf("\nScale=%f zp=%d n_elems=%d\n",
        output_attrs[0].scale, output_attrs[0].zp, output_attrs[0].n_elems);

    printf("Raw -> Dequantized (first 20):\n");
    for (int i=0;i<40;i+=2){
        int8_t q = out[i];
        float f = (q - output_attrs[0].zp) * output_attrs[0].scale;
        printf("%d -> %.6f\n", q, f);
    }
    // 6. Rescale INT8 output back to FP32 and save to file
    if (io_num.n_output > 0) {
        int8_t* out_data_int8 = (int8_t*)output_mems[0]->virt_addr;
        float* out_data_fp32 = (float*)malloc(output_attrs[0].n_elems * sizeof(float));
        int idx = 0;

        if (!out_data_fp32) {
            printf("Failed to allocate memory for FP32 output\n");
            rknn_destroy(ctx);
            return -1;
        }
        for (int i = 0; i < output_attrs[0].n_elems; i++) {
            int8_t q = out_data_int8[i];
            out_data_fp32[i] = (q - output_attrs[0].zp) * output_attrs[0].scale;
        }
    
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



    printf("\n---- Output tensor info ----\n");
    printf("scale = %f\n", output_attrs[0].scale);
    printf("zero point = %d\n", output_attrs[0].zp);
    printf("qnt_type = %d\n", output_attrs[0].qnt_type);
    printf("fl = %d\n", output_attrs[0].fl);


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
