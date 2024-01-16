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

#define NO_KEY -1
#define BLOCK_SIZE 512

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

__global__ void insert_arr(int32_t *arr, uint32_t len, uint32_t* XOR_val, uint32_t* right_shift, uint32_t table_num, uint32_t capacity, int32_t *table, uint32_t evict_bound, bool *need_rehash)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < len)
    {
        uint32_t evict_times = 0;
        uint32_t func_id = 0;
        do
        {
            uint32_t pos = get_hash_res(arr[i], XOR_val[func_id], right_shift[func_id], capacity);
            int32_t old_key = atomicExch(&table[pos + capacity * func_id], arr[i]);
            if(old_key == NO_KEY)
            {
                return;
            }
            else
            {
                arr[i] = old_key;
                func_id = (func_id + 1) % table_num;
                evict_times ++;
            }
        } while (evict_times < evict_bound);   

        *need_rehash = true;
    }
}

__global__ void d_search_arr(int32_t *arr, uint32_t len, uint32_t* XOR_val, uint32_t* right_shift, uint32_t table_num, uint32_t capacity, int32_t *table, bool *res_arr, bool *not_found)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < len)
    {
        uint32_t key_found[table_num] = {0};
        for(auto j = 0UL; j < table_num; j ++)
        {
            uint32_t pos = get_hash_res(arr[i], XOR_val[j], right_shift[j], capacity);
            if(table[pos + capacity * j] == arr[i])
            {
                key_found[j] = 1;
            }
        }
    }
}

class Cuckoo_hash
{
private:
    uint32_t m_capacity;
    uint32_t m_capacity_bits;
    uint32_t m_table_num;
    int32_t *d_table;
    Hash_function m_functions;
    uint32_t *d_fun_XOR_val;
    uint32_t *d_fun_right_shift;

    const uint32_t rehash_bound = 100;

    void rehash()
    {
        m_functions = Hash_function(m_capacity_bits, m_table_num);
        cudaMemcpy(d_fun_right_shift, m_functions.m_right_shift, sizeof(uint32_t) * m_table_num, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fun_XOR_val, m_functions.m_XOR_val, sizeof(uint32_t) * m_table_num, cudaMemcpyHostToDevice);

        cudaMemset(d_table, NO_KEY, sizeof(int32_t) * m_table_num * m_capacity);
    }



public:
    Cuckoo_hash(uint32_t table_size, uint32_t function_num): m_capacity_bits(std::floor(std::log2(table_size))), m_capacity(table_size), m_table_num(function_num), m_functions(m_capacity_bits, function_num)
    {

        cudaMalloc((void **)&d_table, sizeof(int32_t) * m_table_num * m_capacity);
        cudaMemset(d_table, 0, sizeof(uint32_t) * m_capacity * m_table_num);
        
        cudaMalloc((void **)&d_fun_right_shift, sizeof(uint32_t) * m_table_num);
        cudaMalloc((void **)&d_fun_XOR_val, sizeof(uint32_t) * m_table_num);

        cudaMemset(d_table, NO_KEY, sizeof(int32_t) * m_table_num);

        cudaMemcpy(d_fun_right_shift, m_functions.m_right_shift, sizeof(uint32_t) * m_table_num, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fun_XOR_val, m_functions.m_XOR_val, sizeof(uint32_t) * m_table_num, cudaMemcpyHostToDevice);

    }

    ~Cuckoo_hash()
    {
        cudaFree(d_table);
    }

    bool insert(int32_t *arr, uint32_t len)
    {
        int32_t *d_arr;
        cudaMalloc((void**)&d_arr, sizeof(int32_t) * len);
        cudaMemcpy(d_arr, arr, sizeof(int32_t) * len, cudaMemcpyHostToDevice);

        bool need_rehash = false;
        bool *d_need_rehash;
        cudaMalloc((void**)&d_need_rehash, sizeof(bool));

        uint32_t evict_bound = 4 * std::floor(std::log2(len));
        uint32_t rehash_times = 0;

        dim3 block(BLOCK_SIZE);
        dim3 grid((len + block.x - 1) / block.x);
        cudaMemset(d_need_rehash, false, sizeof(bool));
        insert_arr<<<grid, block>>>(d_arr, len, d_fun_XOR_val, d_fun_right_shift, m_table_num, m_capacity, d_table, evict_bound, d_need_rehash);
        cudaMemcpy(&need_rehash, d_need_rehash, sizeof(bool), cudaMemcpyDeviceToHost);

        while (need_rehash && rehash_times < rehash_bound)
        {
            rehash_times ++;
            rehash();
            cudaMemset(d_need_rehash, false, sizeof(bool));
            insert_arr<<<grid, block>>>(d_arr, len, d_fun_XOR_val, d_fun_right_shift, m_table_num, m_capacity, d_table, evict_bound, d_need_rehash);
            cudaMemcpy(&need_rehash, d_need_rehash, sizeof(bool), cudaMemcpyDeviceToHost);
        }
        
        return need_rehash;
    }

    bool 
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
    Cuckoo_hash hash(25, 1 << 2);
    auto len_bit = 2;
    auto len = 1 << len_bit;
    int32_t *arr = (int32_t *)malloc(sizeof(int32_t) * len);
    Random_array(arr, len_bit);
    auto test = hash.insert(arr, len);
    std::cout << "insert " << len << " elements, need rehash: " << test << "\n";
    return 0;
}
