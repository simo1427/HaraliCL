#ifndef WINDOW_SIZE
#error Window size not defined. Pass it as a command-line argument via -DWINDOW_SIZE=.
#endif

typedef struct GrayPair {
    uchar ref, val;
    float weight;
} graypair_t;

int calculate_address(int _ncols, int dx, int dy, uint cnt) {
    int numaddrows = cnt / WINDOW_SIZE, mod = cnt % WINDOW_SIZE;
    return _ncols * (get_group_id(0) * (WINDOW_SIZE) + numaddrows) +
         (get_group_id(1)) * (WINDOW_SIZE) + mod;
}

int searchpair(__global graypair_t *pair, uchar ref, uchar val, int _ncols,
               int dx, int dy, uint cnt) {
    int k;
    for (k = 0; k < cnt; k++) {
    int addr = calculate_address(_ncols, dx, dy, k);
    if (pair[addr].ref == ref && pair[addr].val == val)
      return addr;
    }
    return -1;
}


void clearpairs(__global graypair_t *pair, int _ncols, int dx, int dy,
                 uint cnt) {
    int k;
    for (k = 0; k < cnt; k++) {
        int addr = calculate_address(_ncols, dx, dy, k);
        pair[addr].ref = 0;
        pair[addr].val = 0;
        pair[addr].weight = 0.0f;
    }
    return;
}


__kernel void glcmgen(__global uchar *img, __global float *res,
                                __global graypair_t *pair, int dx, int dy,
                                int nrows, int ncols) {
    int i = get_global_id(0), j = get_global_id(1);
    int counts = (WINDOW_SIZE - dx) * (WINDOW_SIZE - dy);
#ifdef SYMMETRIC
    counts *= 2;
#endif
    int hws = WINDOW_SIZE / 2;
    int xstart = hws, xend = ncols - hws;
    int ystart = hws, yend = nrows - hws;
    float sum_private = 0.0f;
    int stridex = (xend - xstart) / WINDOW_SIZE + 1,
            stridey = (yend - ystart) / WINDOW_SIZE + 1;
    int yrangestart = get_group_id(0) * WINDOW_SIZE,
            yrangeend = (get_group_id(0) + 1) * WINDOW_SIZE;
    int xrangestart = get_group_id(1) * WINDOW_SIZE,
            xrangeend = (get_group_id(1) + 1) * WINDOW_SIZE;
    __local uint cnt;
    __local float meani, meanj, vari, varj;
    float increment;

#ifdef SYMMETRIC
    increment = 2.0f;
#else
    increment = 1.0f;
#endif

    meani = 0.0f;
    meanj = 0.0f;
    vari = 0.0f;
    varj = 0.0f;
    cnt = 0;
    for (i = yrangestart; i < yrangeend; i++) {
        for (j = xrangestart; j < xrangeend; j++) {
            if (i < yend && j < xend && i >= ystart && j >= xstart) {
                for (int imaddr0 = (dy < 0 ? -hws + dy : -hws);
                         imaddr0 <= (dy > 0 ? hws - dy : hws); imaddr0++) {
                    for (int imaddr1 = (dx < 0 ? -hws + dx : -hws);
                             imaddr1 <= (dx > 0 ? hws - dx : hws); imaddr1++) {
                        int ref = img[(i + imaddr0) * ncols + j + imaddr1];
                        int val = img[(i + imaddr0 + dy) * ncols + (imaddr1 + dx + j)];
#ifdef SYMMETRIC
                        int tmp = searchpair(pair, min(ref, val), max(ref, val), (ncols),
                                                                 dx, dy, cnt);
#else
                        int tmp = searchpair(pair, ref, val, (ncols), dx, dy, cnt);
#endif
                        if (tmp == -1) {
                            pair[calculate_address((ncols), dx, dy, cnt)].ref = ref;
                            pair[calculate_address((ncols), dx, dy, cnt)].val = val;
                            pair[calculate_address((ncols), dx, dy, cnt)].weight = increment;
                            cnt++;
                        } else {
                            pair[tmp].weight += increment;
                        }
                    }
                }
                // dissimilarity 0
                for (int pairaddr0 = 0; pairaddr0 < cnt + 1; pairaddr0++) {
                    int addr = calculate_address((ncols), dx, dy, pairaddr0);
                    sum_private += (pair[addr].weight / (counts)) *
                                                 abs(pair[addr].ref - pair[addr].val);
                }
                res[0 * (ncols - 2 * hws) * (nrows - 2 * hws) +
                        (i - hws) * (ncols - 2 * hws) + (j - hws)] = sum_private;
                sum_private = 0.0f;
                // contrast 1
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = calculate_address((ncols), dx, dy, pairaddr0);
                    sum_private += (pair[addr].weight / (counts)) *
                                                 (pair[addr].ref - pair[addr].val) *
                                                 (pair[addr].ref - pair[addr].val);
                }
                res[1 * (ncols - 2 * hws) * (nrows - 2 * hws) +
                        (i - hws) * (ncols - 2 * hws) + (j - hws)] = sum_private;
                sum_private = 0.0f;
                // homogeneity 2
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = calculate_address((ncols), dx, dy, pairaddr0);
                    float tmpres = (pair[addr].weight / (counts)) /
                                                 (1 + (pair[addr].ref - pair[addr].val) *
                                                                    (pair[addr].ref - pair[addr].val));
                    sum_private += (pair[addr].weight / (counts)) /
                                                 (1 + (pair[addr].ref - pair[addr].val) *
                                                                    (pair[addr].ref - pair[addr].val));
                }
                res[2 * (ncols - 2 * hws) * (nrows - 2 * hws) +
                        (i - hws) * (ncols - 2 * hws) + (j - hws)] = sum_private;
                sum_private = 0.0f;
                // ASM 3
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = calculate_address((ncols), dx, dy, pairaddr0);
                    sum_private +=
                            (pair[addr].weight / (counts)) * (pair[addr].weight / counts);
                }
                res[3 * (ncols - 2 * hws) * (nrows - 2 * hws) +
                        (i - hws) * (ncols - 2 * hws) + (j - hws)] = sum_private;
                sum_private = 0.0f;
                // energy 4
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = calculate_address((ncols), dx, dy, pairaddr0);
                    sum_private +=
                            (pair[addr].weight / (counts)) * (pair[addr].weight / counts);
                }
                res[4 * (ncols - 2 * hws) * (nrows - 2 * hws) +
                        (i - hws) * (ncols - 2 * hws) + (j - hws)] = sqrt(sum_private);
                sum_private = 0.0f;
                // entropy 5
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = calculate_address((ncols), dx, dy, pairaddr0);
                    float tmpres = -(pair[addr].weight / (counts)) *
                                                 log((pair[addr].weight / counts));
                    if (isnan(tmpres))
                        continue;
                    sum_private += tmpres;
                }
                res[5 * (ncols - 2 * hws) * (nrows - 2 * hws) +
                        (i - hws) * (ncols - 2 * hws) + (j - hws)] = sum_private;
                sum_private = 0.0f;
                // Mean
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = calculate_address((ncols), dx, dy, pairaddr0);
//#ifdef SYMMETRIC
//                    meani += (pair[addr].weight / (counts)) *
//                                     (pair[addr].ref + pair[addr].val); // symmetrical pairs!
//#else
                    meani += (pair[addr].weight / (counts)) * (pair[addr].ref);
                    meanj += (pair[addr].weight / (counts)) * (pair[addr].val);
//#endif
                }
                // Variance
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = calculate_address((ncols), dx, dy, pairaddr0);
//#ifdef SYMMETRIC
//                    vari += (pair[addr].weight / (counts)) *
//                                    (pow((pair[addr].ref - meani), 2) +
//                                     pow(pair[addr].val - meanj, 2));
//#else
                    vari += (pair[addr].weight / (counts)) *
                                    (pow((pair[addr].ref - meani), 2));
                    varj +=
                            (pair[addr].weight / (counts)) * pow((pair[addr].val - meanj), 2);
//#endif
                }
//#ifdef SYMMETRIC
//                meanj = meani;
//                varj = vari;
//#endif
                // Correlation 6
                for (int pairaddr0 = 0; pairaddr0 < cnt; pairaddr0++) {
                    int addr = calculate_address((ncols), dx, dy, pairaddr0);
                    sum_private +=
                            (pair[addr].weight / (counts)) *
                            (((pair[addr].ref - meani) * (pair[addr].val - meanj)) /
                             (sqrt(vari * varj)));
                }
                res[6 * (ncols - 2 * hws) * (nrows - 2 * hws) +
                        (i - hws) * (ncols - 2 * hws) + (j - hws)] = sum_private;
                meani = 0;
                meanj = 0;
                vari = 0;
                varj = 0;
                sum_private = 0.0f;
                cnt = 0;
            }
        }
    }
}
