#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define NUM_SAMPLES 1000

#define INPUT_LENGTH 2

#define TEMP_MIN 26.7
#define HUM_MIN 40.01
#define TEMP_RANGE 13.3
#define HUM_RANGE 59.97

#define OUTPUT_MAX 105.72
#define OUTPUT_MIN 26.79
#define OUTPUT_RANGE 78.93

float temp = 36.64;
float hum = 80.46;

float W1[16] = {
0.137035f, -0.091562f, -0.556704f, 0.019155f, -0.161019f, -0.017937f, -0.306512f, 0.864718f, -0.330848f, 0.281924f, -0.532878f, -0.341697f, 0.577816f, 0.048100f, -0.169954f, 0.443168f
};

float b1[8] = {
0.247076f, -0.010449f, 0.000000f, -0.021452f, -0.020286f, -0.047487f, 0.000000f, -0.138814f
};

float W2[128] = {
-0.028488f, -0.312762f, 0.339321f, 0.451213f, 0.138466f, -0.460974f, -0.024100f, 0.264286f, -0.435933f, -0.283501f, -0.148148f, -0.233364f, 0.103016f, -0.322032f, -0.479543f, -0.067806f, -0.540677f, 0.227184f, -0.758702f, 0.173562f, -0.553398f, -0.145810f, 0.142089f, -0.612877f, 0.132478f, -0.238455f, 0.421089f, 0.388299f, -0.075838f, -0.211295f, 0.011788f, -0.066206f, 0.368206f, -0.357022f, -0.218328f, -0.426850f, 0.118410f, 0.228729f, -0.101586f, 0.104666f, 0.400147f, -0.005330f, -0.287854f, -0.185351f, -0.460494f, 0.094882f, -0.290713f, -0.418208f, 0.053260f, -0.051573f, 0.106139f, 0.495934f, -0.359359f, 0.325669f, 0.117595f, 0.053897f, -0.368324f, 0.451835f, -0.115408f, -0.147055f, 0.324721f, -0.016215f, 0.463784f, 0.391145f, 0.185924f, -0.489685f, 0.266564f, -0.434082f, 0.268744f, 0.119737f, 0.146368f, 0.272851f, 0.164054f, 0.387266f, -0.256913f, -0.318054f, 0.452810f, 0.331019f, -0.096089f, -0.106197f, -0.332255f, 0.194988f, 0.457046f, 0.422707f, -0.394724f, -0.297444f, 0.128198f, 0.136069f, -0.089134f, 0.233871f, 0.195322f, 0.325205f, -0.398269f, -0.022485f, -0.135040f, -0.134486f, -0.283255f, 0.466350f, 0.423290f, 0.110785f, -0.365049f, -0.231292f, 0.302751f, 0.360567f, -0.391775f, -0.494268f, 0.199426f, -0.091824f, -0.118512f, 0.251472f, -0.229842f, -0.350090f, -0.524862f, -0.291460f, -0.403583f, 0.117931f, -0.655287f, 0.416332f, 0.548073f, -0.392245f, -0.119636f, 0.491278f, 0.357942f, -0.417607f, -0.240608f, -0.274887f, 0.415069f, 0.306656f
};

float b2[16] = {
0.036539f, 0.000000f, 0.029982f, -0.027447f, 0.051245f, -0.360224f, -0.074321f, 0.258884f, -0.089330f, -0.087721f, -0.294047f, 0.000000f, 0.154418f, -0.050867f, -0.117148f, -0.015506f
};

float W3[16] = {
-0.471396f, 0.465381f, -0.288610f, -0.324588f, -0.476918f, 0.305728f, 0.386841f, 0.894563f, 0.215420f, 0.508898f, 0.332013f, -0.327066f, -0.541701f, 0.446995f, 0.725584f, 0.877572f
};

float b3[1] = {
-0.067937f
};

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}


void dense(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int in_dim,
    int out_dim
) {
    for (int o = 0; o < out_dim; o++) {
        float acc = bias[o];
        for (int i = 0; i < in_dim; i++) {
            acc += input[i] * weights[i * out_dim + o];
        }
        output[o] = acc;
    }
}

void model_forward(const float input[2], float* output) {
    float l1[8];
    float l2[16];
    float l3;

    // Layer 1
    dense(input, W1, b1, l1, 2, 8);
    for (int i = 0; i < 8; i++)
        l1[i] = relu(l1[i]);

    // Layer 2
    dense(l1, W2, b2, l2, 8, 16);
    for (int i = 0; i < 16; i++)
        l2[i] = relu(l2[i]);

    // Layer 3
    dense(l2, W3, b3, &l3, 16, 1);

    *output = l3;
}


static inline double time_diff_us(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1000000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000.0;
}


int main(int argc, char **argv)
{

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

    FILE *fout = fopen("output_model.txt", "w");
    if (!fout)
    {
        printf("Failed to open output.txt\n");
        return -1;
    }

    fprintf(fout, "Temp,Hum,Output,Time_us\n");

    // Batch inference
    for (int i = 0; i < count; i++)
    {
        float y;
        float scaled_sample[2];
        float temp = samples[i][0];  // Convert to Fahrenheit
        float hum = samples[i][1];

        scaled_sample[0] = (temp - TEMP_MIN)/TEMP_RANGE;
        scaled_sample[1] = (hum - HUM_MIN)/HUM_RANGE;

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        model_forward(scaled_sample,&y);

        clock_gettime(CLOCK_MONOTONIC, &end);

        double time_us = time_diff_us(start, end);
        float final = y*OUTPUT_RANGE + OUTPUT_MIN;

        // Save to file
        fprintf(fout, "%.2f,%.2f,%.5f,%.2f\n",
                samples[i][0], hum, final, time_us);

        if (i % 100 == 0)
            printf("Processed %d / %d\n", i, count);
    }

    fclose(fout);

    return 0;
}
