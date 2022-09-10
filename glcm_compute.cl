#ifndef WINDOW_SIZE
#error Window size not defined. Pass it as a command-line argument via -DWINDOW_SIZE=.
#endif

/**
 * A struct type that stores the information found in a pair = the two
 * neighbouring pixels and the weight of that neighbourhood.
 */
typedef struct GrayPair {
    uchar ref, val;
    float weight;
} graypair_t;

/**
 * Searches for the pair in the local memory. Returns -1 if no such pair is yet stored.
 * @param pair the array that stores pairs
 * @param ref the first value of the GLCM pair (reference)
 * @param val the second value of the GLCM pair (value)
 * @param cnt number of times this pair is found in the sliding window
 * @return the address of the pair in the given array
 */
int searchpair(__local graypair_t *pair, uchar ref, uchar val, uint cnt) {
    for (int k = 0; k < cnt; k++) {
        if (pair[k].ref == ref && pair[k].val == val)
          return k;
    }
    return -1;
}

/**
 * Initialize empty values for storing pair information.
 * @param pair the array that stores pairs
 * @param cnt the number of entries to clear
 */
void clearpairs(__local graypair_t *pair, uint cnt) {
    for (int k = 0; k < cnt; k++) {
        pair[k].ref = 0;
        pair[k].val = 0;
        pair[k].weight = 0.0f;
    }
    return;
}

/**
 * Computes sliding-window based on Haralick features.
 * @param img the image as flattened uchar aray
 * @param res the final computed images
 * @param dx offset on the x-axis (can be negative)
 * @param dy offset on the y-axis (can be negative)
 */
__kernel void glcmgen(__global uchar *img, __global float *res,
                                int dx, int dy,
                                int rows, int cols) {
    int i = get_global_id(0), j = get_global_id(1);
    int counts = (WINDOW_SIZE - dx) * (WINDOW_SIZE - dy);
#ifdef SYMMETRIC
    counts *= 2;
#endif
    int half_window_size = WINDOW_SIZE / 2;
    int x_start = half_window_size, x_end = cols - half_window_size;
    int y_start = half_window_size, y_end = rows - half_window_size;
    float sum_private = 0.0f;
    int stridex = (x_end - x_start) / WINDOW_SIZE + 1,
            stridey = (y_end - y_start) / WINDOW_SIZE + 1;
    int yrangestart = get_group_id(0) * WINDOW_SIZE,
            yrangeend = (get_group_id(0) + 1) * WINDOW_SIZE;
    int xrangestart = get_group_id(1) * WINDOW_SIZE,
            xrangeend = (get_group_id(1) + 1) * WINDOW_SIZE;
    __local uint cnt;
    __local float mean_i, mean_j, variance_i, variance_j;
    float increment;

#ifdef SYMMETRIC
    increment = 2.0f;
#else
    increment = 1.0f;
#endif

    __local graypair_t pair[WINDOW_SIZE*WINDOW_SIZE];
    clearpairs(pair, WINDOW_SIZE*WINDOW_SIZE);

    mean_i = 0.0f;
    mean_j = 0.0f;
    variance_i = 0.0f;
    variance_j = 0.0f;
    cnt = 0;
    for (i = yrangestart; i < yrangeend; i++) {
        for (j = xrangestart; j < xrangeend; j++) {
            if (i < y_end && j < x_end && i >= y_start && j >= x_start) {
                for (int imaddr0 = (dy < 0 ? -half_window_size + dy : -half_window_size);
                         imaddr0 <= (dy > 0 ? half_window_size - dy : half_window_size); imaddr0++) {
                    for (int imaddr1 = (dx < 0 ? -half_window_size + dx : -half_window_size);
                             imaddr1 <= (dx > 0 ? half_window_size - dx : half_window_size); imaddr1++) {
                        int ref = img[(i + imaddr0) * cols + j + imaddr1];
                        int val = img[(i + imaddr0 + dy) * cols + (imaddr1 + dx + j)];
#ifdef SYMMETRIC
                        int tmp = searchpair(pair, min(ref, val), max(ref, val), cnt);
#else
                        int tmp = searchpair(pair, ref, val, cnt);
#endif
                        if (tmp == -1) {
                            pair[cnt].ref = ref;
                            pair[cnt].val = val;
                            pair[cnt].weight = increment;
                            cnt++;
                        } else {
                            pair[tmp].weight += increment;
                        }
                    }
                }
                // dissimilarity 0
                for (int pairaddr0 = 0; pairaddr0 < cnt + 1; pairaddr0++) {
                    int addr = pairaddr0;
                    sum_private += (pair[addr].weight / (counts)) *
                                                 abs(pair[addr].ref - pair[addr].val);
                }
                res[0 * (cols - 2 * half_window_size) * (rows - 2 * half_window_size) +
                        (i - half_window_size) * (cols - 2 * half_window_size) + (j - half_window_size)] = sum_private;
                sum_private = 0.0f;
                // contrast 1
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = pairaddr0;
                    sum_private += (pair[addr].weight / (counts)) *
                                                 (pair[addr].ref - pair[addr].val) *
                                                 (pair[addr].ref - pair[addr].val);
                }
                res[1 * (cols - 2 * half_window_size) * (rows - 2 * half_window_size) +
                        (i - half_window_size) * (cols - 2 * half_window_size) + (j - half_window_size)] = sum_private;
                sum_private = 0.0f;
                // homogeneity 2
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = pairaddr0;
                    float tmpres = (pair[addr].weight / (counts)) /
                                                 (1 + (pair[addr].ref - pair[addr].val) *
                                                                    (pair[addr].ref - pair[addr].val));
                    sum_private += (pair[addr].weight / (counts)) /
                                                 (1 + (pair[addr].ref - pair[addr].val) *
                                                                    (pair[addr].ref - pair[addr].val));
                }
                res[2 * (cols - 2 * half_window_size) * (rows - 2 * half_window_size) +
                        (i - half_window_size) * (cols - 2 * half_window_size) + (j - half_window_size)] = sum_private;
                sum_private = 0.0f;
                // ASM 3
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = pairaddr0;
                    sum_private +=
                            (pair[addr].weight / (counts)) * (pair[addr].weight / counts);
                }
                res[3 * (cols - 2 * half_window_size) * (rows - 2 * half_window_size) +
                        (i - half_window_size) * (cols - 2 * half_window_size) + (j - half_window_size)] = sum_private;
                sum_private = 0.0f;
                // energy 4
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = pairaddr0;
                    sum_private +=
                            (pair[addr].weight / (counts)) * (pair[addr].weight / counts);
                }
                res[4 * (cols - 2 * half_window_size) * (rows - 2 * half_window_size) +
                        (i - half_window_size) * (cols - 2 * half_window_size) + (j - half_window_size)] = sqrt(sum_private);
                sum_private = 0.0f;
                // entropy 5
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = pairaddr0;
                    float tmpres = -(pair[addr].weight / (counts)) *
                                                 log((pair[addr].weight / counts));
                    if (isnan(tmpres))
                        continue;
                    sum_private += tmpres;
                }
                res[5 * (cols - 2 * half_window_size) * (rows - 2 * half_window_size) +
                        (i - half_window_size) * (cols - 2 * half_window_size) + (j - half_window_size)] = sum_private;
                sum_private = 0.0f;
                // Mean
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = pairaddr0;
                    mean_i += (pair[addr].weight / (counts)) * (pair[addr].ref);
                    mean_j += (pair[addr].weight / (counts)) * (pair[addr].val);
                }
                // variance_iance
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = pairaddr0;
                    variance_i += (pair[addr].weight / (counts)) *
                                    (pow((pair[addr].ref - mean_i), 2));
                    variance_j +=
                            (pair[addr].weight / (counts)) * pow((pair[addr].val - mean_j), 2);
                }
                // Correlation 6
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = pairaddr0;
                    sum_private +=
                            (pair[addr].weight / (counts)) *
                            (((pair[addr].ref - mean_i) * (pair[addr].val - mean_j)) /
                             (sqrt(variance_i * variance_j)));
                }
                res[6 * (cols - 2 * half_window_size) * (rows - 2 * half_window_size) +
                        (i - half_window_size) * (cols - 2 * half_window_size) + (j - half_window_size)] = sum_private;
                mean_i = 0;
                mean_j = 0;
                variance_i = 0;
                variance_j = 0;
                sum_private = 0.0f;
                cnt = 0;
            }
        }
    }
}
