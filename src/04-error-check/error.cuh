#pragma once
#include <stdio.h>

// #pragma once是预处理指令，作用是确保当前文件在一个编译单元中不被重复包含
// 下面定义了一个宏函数，名称是CHECK
// 在定义宏的时候一行写不下需要换行，在行末加上'\'
// 所有CUDA运行时api的返回值都是cudaError_t类型，当返回值为cudaSuccess时代表调用成功
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

