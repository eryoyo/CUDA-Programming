#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif

const int NUM_REPEATS = 10;
const real a = 1.23;
const real b = 2.34;
const real c = 3.57;
void __global__ add(const real *x, const real *y, real *z, const int N);
void check(const real *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *h_x = (real*) malloc(M);
    real *h_y = (real*) malloc(M);
    real *h_z = (real*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void **)&d_x, M));
    CHECK(cudaMalloc((void **)&d_y, M));
    CHECK(cudaMalloc((void **)&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        // cuda事件类型
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        // 对于处于TCC驱动模式的GPU可以省略，但是对于WDDM驱动模式下的GPU必须保留
        // 因为WDDM模式下一个CUDA stream中的操作，例如此处的cudaEventRecord函数不是直接交给GPU执行，而是先提交到一个软件队列
        // 需要添加一条对该流的 cudaEventQuery 操作（或者cudaEventSynchronize）刷新队列，才能促使前面的操作在GPU执行
        cudaEventQuery(start);

        add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        // 计算两个事件之间的时间差
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

void __global__ add(const real *x, const real *y, real *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}

void check(const real *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

