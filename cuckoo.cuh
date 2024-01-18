#ifndef CUCKOO_HASH_HPP
#define CUCKOO_HASH_HPP

#include <iostream>
#include <cuda.h>
#include <vector>
#include <random>
#include <limits>
#include <cctype>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#define NO_KEY -1
#define BLOCK_SIZE 512
#define REPEAT_TIME 5
#define GET_PERFOEMANCE(data_size, seconds) (data_size / seconds / 1e6)
#define REHASH_BOUND 100
#define MAX_FUNCTION 10

#define PER_BLOCK 1

// #define MODIFY

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

    void rehash()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0, (uint32_t)(-1));
        std::uniform_int_distribution<uint32_t> dis2(0, 30 - m_capacity_bits);

        for(auto i = 0UL; i < m_function_num; i ++)
        {
            m_XOR_val[i] = dis(gen);
            m_right_shift[i] = dis2(gen);
        }
    }

    ~Hash_function()
    {
        free(m_XOR_val);
        free(m_right_shift);
    }

friend class Cuckoo_hash;
};

__forceinline__ __device__ uint32_t get_hash_res(int32_t input, uint32_t XOR_val, uint32_t right_shift, uint32_t capacity)
{
    return ((input ^ XOR_val) >> right_shift) % capacity;
}

__global__ void d_insert_arr(const int32_t *arr, const uint32_t len, const uint32_t* XOR_val, const uint32_t* right_shift, const uint32_t table_num, const uint32_t function_num,const uint32_t capacity, int32_t *table,const uint32_t evict_bound, bool *need_rehash)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t XOR_val_local[MAX_FUNCTION];
    uint32_t right_shift_local[MAX_FUNCTION];
    for(auto j = 0; j < function_num; j ++)
    {
        XOR_val_local[j] = XOR_val[j];
        right_shift_local[j] = right_shift[j];
    }
    if(i < len)
    {
        uint32_t evict_times = 0;
        uint32_t func_id = 0;
        uint32_t table_id = 0;
        int32_t key = arr[i];
        bool need_evict = true;
        while(need_evict == true && evict_times < evict_bound * table_num && *need_rehash == false)
        {
            uint32_t pos = get_hash_res(key, XOR_val_local[func_id], right_shift_local[func_id], capacity);
            key = atomicExch(&table[pos + capacity * table_id], key);
            if(key == NO_KEY)
            {
                break;
            }
            else
            {
                func_id = (func_id + 1) % function_num;
                table_id = (table_id + 1) % table_num;
                evict_times ++;
            }
        }

        if(evict_times == evict_bound * table_num)
        {
            *need_rehash = true;
        }
    }
}

__global__ void d_insert_modify(int32_t *arr, int32_t *buffer, uint32_t len, uint32_t *buffer_len, uint32_t* XOR_val, uint32_t* right_shift, uint32_t table_num, uint32_t capacity, int32_t *table)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int32_t old_key_buffer[BLOCK_SIZE];
    if(i < len)
    {
        int32_t old_key = arr[i];
        for(auto j = 0; j < table_num; j ++)
        {
            uint32_t pos = get_hash_res(old_key, XOR_val[j], right_shift[j], capacity);
            old_key = atomicExch(&table[pos + capacity * j], old_key);
            if(old_key == NO_KEY)
            {
                return;
            }
        }
        auto insert_pos = atomicInc(buffer_len, 1);
        buffer[insert_pos] = old_key;
    }
}

__global__ void d_search_arr(int32_t *arr, uint32_t len, uint32_t* XOR_val, uint32_t* right_shift, uint32_t table_num, uint32_t function_num,uint32_t capacity, int32_t *table, bool *key_found)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < len)
    {
        auto to_find = arr[i];
        auto func_id = 0UL;
        for(auto j = 0UL; j < table_num; j ++)
        {
            uint32_t pos = get_hash_res(to_find, XOR_val[func_id], right_shift[func_id], capacity);
            if(table[pos + capacity * j] == to_find)
            {
                key_found[i] = true;
            }
            else
            {
                func_id = (func_id + 1) % function_num;
            }
        }
    }
}

class Cuckoo_hash
{
private:
    uint32_t m_table_num;
    uint32_t m_capacity;
    uint32_t m_capacity_bits;
    int32_t *d_table;
    Hash_function m_functions;
    uint32_t *d_fun_XOR_val;
    uint32_t *d_fun_right_shift;

    const uint32_t rehash_bound = REHASH_BOUND;

    void rehash()
    {
        m_functions.rehash();
        cudaMemcpy(d_fun_right_shift, m_functions.m_right_shift, sizeof(uint32_t) * m_functions.m_function_num, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fun_XOR_val, m_functions.m_XOR_val, sizeof(uint32_t) * m_functions.m_function_num, cudaMemcpyHostToDevice);

        cudaMemset(d_table, NO_KEY, sizeof(int32_t) * m_table_num * m_capacity);
    }



public:
    Cuckoo_hash(uint32_t table_size, uint32_t function_num): m_capacity_bits((uint32_t)std::floor(std::log2(table_size/m_table_num))), m_capacity(table_size/m_table_num), m_table_num(function_num * 2), m_functions(m_capacity_bits, function_num)
    {
        cudaMalloc((void **)&d_table, sizeof(int32_t) * m_table_num * m_capacity);
        cudaMemset(d_table, 0, sizeof(uint32_t) * m_capacity * m_table_num);
        
        cudaMalloc((void **)&d_fun_right_shift, sizeof(uint32_t) * m_functions.m_function_num);
        cudaMalloc((void **)&d_fun_XOR_val, sizeof(uint32_t) * m_functions.m_function_num);

        cudaMemset(d_table, NO_KEY, sizeof(int32_t) * m_table_num * m_capacity);

        cudaMemcpy(d_fun_right_shift, m_functions.m_right_shift, sizeof(uint32_t) * m_functions.m_function_num, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fun_XOR_val, m_functions.m_XOR_val, sizeof(uint32_t) * m_functions.m_function_num, cudaMemcpyHostToDevice);

    }

    ~Cuckoo_hash()
    {
        cudaFree(d_table);
        cudaFree(d_fun_right_shift);
        cudaFree(d_fun_XOR_val);
    }

#ifndef MODIFY
    uint32_t insert(int32_t *arr, uint32_t len, uint32_t evict_bound = 0)
    {
        int32_t *d_arr;
        cudaMalloc((void**)&d_arr, sizeof(int32_t) * len);
        cudaMemcpy(d_arr, arr, sizeof(int32_t) * len, cudaMemcpyHostToDevice);

        bool need_rehash = false;
        bool *d_need_rehash;
        cudaMalloc((void**)&d_need_rehash, sizeof(bool));

        if(evict_bound == 0)
        {
            evict_bound = 4 * std::floor(std::log2(len));
        }
        uint32_t rehash_times = 0;

        dim3 block(BLOCK_SIZE);
        dim3 grid((len + block.x - 1) / block.x / PER_BLOCK);
        cudaMemset(d_need_rehash, false, sizeof(bool));
        d_insert_arr<<<grid, block>>>(d_arr, len, d_fun_XOR_val, d_fun_right_shift, m_table_num, m_functions.m_function_num,m_capacity, d_table, evict_bound, d_need_rehash);
        // cudaDeviceSynchronize();
        cudaMemcpy(&need_rehash, d_need_rehash, sizeof(bool), cudaMemcpyDeviceToHost);

        while (need_rehash && rehash_times < rehash_bound)
        {
            rehash_times ++;
            rehash();
            cudaMemset(d_need_rehash, false, sizeof(bool));
            d_insert_arr<<<grid, block>>>(d_arr, len, d_fun_XOR_val, d_fun_right_shift, m_table_num, m_functions.m_function_num ,m_capacity, d_table, evict_bound, d_need_rehash);
            // cudaDeviceSynchronize();
            cudaMemcpy(&need_rehash, d_need_rehash, sizeof(bool), cudaMemcpyDeviceToHost);
        }
        
        cudaFree(d_arr);
        cudaFree(d_need_rehash);

        // return !need_rehash && rehash_times < rehash_bound;
        return rehash_times;
    }

#else

    uint32_t insert(int32_t *arr, uint32_t len, uint32_t evict_bound = 0)
    {
        int32_t *d_arr;
        cudaMalloc((void**)&d_arr, sizeof(int32_t) * len);
        cudaMemcpy(d_arr, arr, sizeof(int32_t) * len, cudaMemcpyHostToDevice);

        uint32_t buffer_len = len;
        uint32_t *d_buffer_len;
        cudaMalloc((void**)&d_buffer_len, sizeof(uint32_t));
        int32_t *d_buffer;
        cudaMalloc((void **)&d_buffer, sizeof(int32_t) * len);

        if(evict_bound == 0)
        {
            evict_bound = 4 * std::floor(std::log2(len));
        }
        uint32_t evict_times = 0;
        uint32_t rehash_times = 0;

        while(rehash_times < rehash_bound)
        {
            do
            {
                cudaMemset(d_buffer_len, 0, sizeof(uint32_t));
                dim3 block(BLOCK_SIZE);
                dim3 grid((buffer_len + block.x - 1) / block.x);
                d_insert_modify<<<grid, block>>>(d_arr, d_buffer, buffer_len, d_buffer_len, d_fun_XOR_val, d_fun_right_shift, m_table_num, m_capacity, d_table);
                cudaMemcpy(d_buffer_len, &buffer_len, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                std::swap(d_arr, d_buffer);
                evict_times ++;
            } while (buffer_len != 0 && evict_times < evict_bound);

            if(evict_times == evict_bound)
            {
                rehash_times ++;
                rehash();
            }
            else
            {
                break;
            }
        }

        return rehash_times;        

    }
#endif
    bool* search(int32_t *arr, uint32_t len)
    {
        int32_t *d_arr;
        cudaMalloc((void **)&d_arr, sizeof(int32_t) * len);
        cudaMemcpy(d_arr, arr, sizeof(int32_t) * len, cudaMemcpyHostToDevice);

        bool *key_found = (bool *)malloc(sizeof(bool) * len);
        bool *d_key_found;
        cudaMalloc((void **)&d_key_found, sizeof(bool) * len);
        cudaMemset(d_key_found, 0, sizeof(bool) * len);

        dim3 block(BLOCK_SIZE);
        dim3 grid((len + block.x - 1) / block.x);

        d_search_arr<<<grid, block>>>(d_arr, len, d_fun_XOR_val, d_fun_right_shift, m_table_num, m_functions.m_function_num,m_capacity, d_table, d_key_found);

        cudaMemcpy(key_found, d_key_found, sizeof(bool) * len, cudaMemcpyDeviceToHost);
        cudaFree(d_key_found);
        cudaFree(d_arr);

        return key_found;
    }

    void print_table()
    {
        int32_t *table = (int32_t *)malloc(sizeof(int32_t) * m_table_num * m_capacity);
        cudaMemcpy(table, d_table, sizeof(int32_t) * m_table_num * m_capacity, cudaMemcpyDeviceToHost);

        for(auto i = 0; i < m_table_num; i ++)
        {
            std::cout << "table " << i << " info" << std::endl;
            std::cout << "XOR_val = " << m_functions.m_XOR_val[i] << " right shift = " << m_functions.m_right_shift[i] << std::endl;
            for(auto j = 0; j < m_capacity; j ++)
            {
                std::cout << "table index " << j << " value = " << table[i * m_capacity + j] << std::endl;
            }
            std::cout << std::endl;
        }

        free(table);
    }
};



void Random_array(int32_t *dst, uint32_t size)
{
    uint32_t len = size;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dis(std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
    for(uint32_t i = 0; i < len; i ++)
    {
        dst[i] = dis(gen);
    }
}

void get_Random_arr_from_arr(int32_t *src, int32_t *dst, uint32_t src_len, uint32_t dst_len)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dis(0, src_len-1);

    for(auto i = 0; i < dst_len; i ++)
    {
        dst[i] = src[dis(gen)];
    }
}

#endif