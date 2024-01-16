#include <iostream>
#include <cuda.h>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <limits>
#include <cctype>
#include <stdlib.h>
#include <string.h>

class Cuckoo_hash;

class Hash_function
{
private:
    uint32_t m_function_num;
    uint32_t *m_XOR_val;
    uint32_t *m_right_shift;
    uint32_t m_capacity_bits;

public:
    Hash_function(uint32_t capacity_bits, uint32_t function_num) : m_function_num(function_num), m_capacity_bits(capacity_bits)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0, (uint32_t)(-1));
        std::uniform_int_distribution<uint32_t> dis2(0, 32 - capacity_bits);
        m_XOR_val = (uint32_t *)malloc(sizeof(uint32_t) * function_num);
        m_right_shift = (uint32_t *)malloc(sizeof(uint32_t) * function_num);

        for(auto i = 0UL; i < function_num; i ++)
        {
            m_XOR_val[i] = dis(gen);
            m_right_shift[i] = dis2(gen);
        }
    }

friend class Cuckoo_hash;
};

__forceinline__ __device__ uint32_t get_hash_res(int32_t input, uint32_t XOR_val, uint32_t right_shift, uint32_t capacity)
{
    return (((input * input) ^ XOR_val) >> right_shift) % capacity;
}

__global__ void get_hash(int32_t *arr, uint32_t len, uint32_t* XOR_val, uint32_t* right_shift, uint32_t table_num, uint32_t capacity)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < len)
    {
        for(auto j = 0; j < table_num; j++)
        {
            printf("arr[%d] hashed by func %d get %u\n", i, j, get_hash_res(arr[i], XOR_val[j], right_shift[j], capacity));
        }
        __syncthreads();
    }
}

__global__ 

class Cuckoo_hash
{
private:
    uint32_t m_capacity_bits;
    uint32_t m_capacity;
    uint32_t m_table_num;
    uint32_t *d_table;
    Hash_function m_functions;
    uint32_t *d_fun_XOR_val;
    uint32_t *d_fun_right_shift;


public:
    Cuckoo_hash(uint32_t table_size_bit, uint32_t function_num): m_capacity_bits(table_size_bit), m_capacity(1 << table_size_bit), m_table_num(function_num), m_functions(m_capacity_bits, function_num)
    {

        cudaMalloc((void **)&d_table, sizeof(uint32_t) * m_table_num * m_capacity);
        cudaMemset(d_table, 0, sizeof(uint32_t) * m_capacity * m_table_num);
        
        cudaMalloc((void **)&d_fun_right_shift, sizeof(uint32_t) * m_table_num);
        cudaMalloc((void **)&d_fun_XOR_val, sizeof(uint32_t) * m_table_num);

        cudaMemcpy(d_fun_right_shift, m_functions.m_right_shift, sizeof(uint32_t) * m_table_num, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fun_XOR_val, m_functions.m_XOR_val, sizeof(uint32_t) * m_table_num, cudaMemcpyHostToDevice);


    }

    ~Cuckoo_hash()
    {
        cudaFree(d_table);
    }

    void test(int32_t *arr, uint32_t len)
    {
        int32_t *d_arr;
        cudaMalloc((void**)&d_arr, sizeof(int32_t) * len);
        cudaMemcpy(d_arr, arr, sizeof(int32_t) * len, cudaMemcpyHostToDevice);

        get_hash<<<1, len>>>(d_arr, len, d_fun_XOR_val, d_fun_right_shift, m_table_num, m_capacity);   
    }
};



void Random_array(int32_t *dst, uint32_t size_bits)
{
    uint32_t len = 1 << size_bits;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dis(std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
    for(uint32_t i = 0; i < len; i ++)
    {
        dst[i] = dis(gen);
    }
}

int main(int argc, char const *argv[])
{
    Cuckoo_hash hash(25, 2);
    auto len_bit = 2;
    auto len = 1 << len_bit;
    int32_t *arr = (int32_t *)malloc(sizeof(int32_t) * len);
    Random_array(arr, len_bit);
    hash.test(arr, len);
    return 0;
}
