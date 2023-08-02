#include <stdio.h>

// 核函数与主机函数的区别 需要用限定词__global__修饰，返回需要是空类型
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    // 所有线程块组成一个grid，线程块的数目是grid size，为括号中第一个数字的含义
    // 第二个数字是每一个线程块中的线程数目，称为block size
    hello_from_gpu<<<1, 1>>>();
    // 调用输出函数的时候，输出流先存放在缓冲区中，不会自动刷新，需要进行同步操作之后完成刷新
    cudaDeviceSynchronize();
    return 0;
}

