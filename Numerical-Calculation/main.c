#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define NUM_SAMPLES 1000

#define INPUT_LENGTH 2

static inline double time_diff_us(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1000000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000.0;
}

static inline float heat_index(float T, float RH)
{
    float HI = (-42.379 + 2.04901523 * T + 10.14333127 * RH - 0.22475541 * T * RH - 0.00683783 * T * T - 0.05481717 * RH * RH + 0.00122874 * T * T * RH + 0.00085282 * T * RH * RH - 0.00000199 * T * T * RH * RH);

    return HI;
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

    FILE *fout = fopen("output_normal.txt", "w");
    if (!fout)
    {
        printf("Failed to open output.txt\n");
        return -1;
    }

    fprintf(fout, "Temp,Hum,Output,Time_us\n");

    // Calculate heat index and measure time

    // Batch inference
    for (int i = 0; i < count; i++)
    {
        float temp = samples[i][0]*9/5 +32;  // Convert to Fahrenheit
        float hum = samples[i][1];

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        float hi = heat_index(temp, hum);

        clock_gettime(CLOCK_MONOTONIC, &end);

        double time_us = time_diff_us(start, end);

        // Save to file
        fprintf(fout, "%.2f,%.2f,%.5f,%.2f\n",
                samples[i][0], hum, hi, time_us);

        if (i % 100 == 0)
            printf("Processed %d / %d\n", i, count);
    }

    fclose(fout);

    return 0;
}
