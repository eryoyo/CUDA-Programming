#include <stdio.h>

// 网格大小在x、y和z这3个方向的最大允许值分别为2^31 − 1、65535 和65535；
// 线程块大小在x、y和z这3个方向的最大允许值分别为1024、1024 和64。另外还要求线程块总的大小，
// 即blockDim.x、blockDim.y 和blockDim.z 的乘积不能大于1024。也就是说，
// 不管如何定义，一个线程块最多只能有1024个线程。
__global__ void hello_from_gpu()
{
    // gridDim, blockDim都是dim3类型的变量，有x, y, z三个变量
    // blockIdx, threadIdx是uint3类型的变量，也有x, y, z三个变量
    // 多维的网格和线程块本质上还是1维的，一个多维线程指标threadIdx.x、threadIdx.y、threadIdx.z 对应的一维指标为
    // int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
    // 与一个多维线程块指标blockIdx.x、blockIdx.y、blockIdx.z 对应的一维指标没有唯一的定义（主要是因为各个线程块的执行是相互独立的）
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello World from block-%d and thread-(%d, %d)!\n", b, tx, ty);
}

int main(void)
{
    // 一个线程块中的线程还可以细分为不同的线程束（thread warp）。一个线程束（即一束线程）是同一个线程块中相邻的warpSize个线程。
    // warpSize 也是一个内建变量，表示线程束大小，其值对于目前所有的GPU 架构都是32。所以，一个线程束就是连续的32个线程。具体地说，
    // 一个线程块中第0到第31个线程属于第0个线程束，第32到第63个线程属于第1个线程束，依此类推。
    const dim3 block_size(2, 4);
    hello_from_gpu<<<1, block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}

