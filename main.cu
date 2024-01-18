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
#include <chrono>
#include "cuckoo.cuh"

#define NO_KEY -1
#define BLOCK_SIZE 4
#define REPEAT_TIME 5
#define GET_PERFOEMANCE(data_size, seconds) (data_size / seconds / 1e6)

// #define MODIFY

// class Cuckoo_hash;

// class Hash_function
// {
// private:
//     uint32_t m_function_num;
//     uint32_t *m_XOR_val;
//     uint32_t *m_right_shift;
//     uint32_t m_capacity_bits;

// public:
//     Hash_function(uint32_t capacity_bits, uint32_t function_num) : m_function_num(function_num), m_capacity_bits(capacity_bits)
//     {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_int_distribution<uint32_t> dis(0, (uint32_t)(-1));
//         std::uniform_int_distribution<uint32_t> dis2(0, 32 - capacity_bits);
//         m_XOR_val = (uint32_t *)malloc(sizeof(uint32_t) * function_num);
//         m_right_shift = (uint32_t *)malloc(sizeof(uint32_t) * function_num);

//         for(auto i = 0UL; i < function_num; i ++)
//         {
//             m_XOR_val[i] = dis(gen);
//             m_right_shift[i] = dis2(gen);
//         }
//     }

//     void rehash()
//     {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_int_distribution<uint32_t> dis(0, (uint32_t)(-1));
//         std::uniform_int_distribution<uint32_t> dis2(0, 30 - m_capacity_bits);

//         for(auto i = 0UL; i < m_function_num; i ++)
//         {
//             m_XOR_val[i] = dis(gen);
//             m_right_shift[i] = dis2(gen);
//         }
//     }

//     ~Hash_function()
//     {
//         free(m_XOR_val);
//         free(m_right_shift);
//     }

// friend class Cuckoo_hash;
// };

// __forceinline__ __device__ uint32_t get_hash_res(int32_t input, uint32_t XOR_val, uint32_t right_shift, uint32_t capacity)
// {
//     return ((input ^ XOR_val) >> right_shift) % capacity;
// }

// __global__ void d_insert_arr(const int32_t *arr, const uint32_t len, const uint32_t* XOR_val, const uint32_t* right_shift, const uint32_t table_num,const uint32_t capacity, int32_t *table,const uint32_t evict_bound, bool *need_rehash)
// {
//     auto i = blockDim.x * blockIdx.x + threadIdx.x;
//     if(i < len)
//     {
//         uint32_t evict_times = 0;
//         uint32_t func_id = 0;
//         int32_t key = arr[i];
//         bool need_evict = true;
//         while(need_evict == true && evict_times < evict_bound * table_num)
//         {
//             uint32_t pos = get_hash_res(key, XOR_val[func_id], right_shift[func_id], capacity);
//             key = atomicExch(&table[pos + capacity * func_id], key);
//             if(key == NO_KEY)
//             {
//                 break;
//             }
//             else
//             {
//                 func_id = (func_id + 1) % table_num;
//                 evict_times ++;
//             }
//         }

//         if(evict_times == evict_bound * table_num)
//         {
//             *need_rehash = true;
//         }
//     }
// }

// __global__ void d_insert_modify(int32_t *arr, int32_t *buffer, uint32_t len, uint32_t *buffer_len, uint32_t* XOR_val, uint32_t* right_shift, uint32_t table_num, uint32_t capacity, int32_t *table)
// {
//     auto i = blockDim.x * blockIdx.x + threadIdx.x;
//     if(i < len)
//     {
//         int32_t old_key = arr[i];
//         for(auto j = 0; j < table_num; j ++)
//         {
//             uint32_t pos = get_hash_res(old_key, XOR_val[j], right_shift[j], capacity);
//             old_key = atomicExch(&table[pos + capacity * j], old_key);
//             if(old_key == NO_KEY)
//             {
//                 return;
//             }
//         }
//         auto insert_pos = atomicInc(buffer_len, 1);
//         buffer[insert_pos] = old_key;
//     }
// }

// __global__ void d_search_arr(int32_t *arr, uint32_t len, uint32_t* XOR_val, uint32_t* right_shift, uint32_t table_num, uint32_t capacity, int32_t *table, bool *key_found)
// {
//     auto i = blockDim.x * blockIdx.x + threadIdx.x;
//     if(i < len)
//     {
//         auto to_find = arr[i];
//         for(auto j = 0UL; j < table_num; j ++)
//         {
//             uint32_t pos = get_hash_res(to_find, XOR_val[j], right_shift[j], capacity);
//             if(table[pos + capacity * j] == to_find)
//             {
//                 key_found[i] = true;
//             }
//         }
//     }
// }

// class Cuckoo_hash
// {
// private:
//     uint32_t m_capacity;
//     uint32_t m_capacity_bits;
//     uint32_t m_table_num;
//     int32_t *d_table;
//     Hash_function m_functions;
//     uint32_t *d_fun_XOR_val;
//     uint32_t *d_fun_right_shift;

//     const uint32_t rehash_bound = 200;

//     void rehash()
//     {
//         m_functions.rehash();
//         cudaMemcpy(d_fun_right_shift, m_functions.m_right_shift, sizeof(uint32_t) * m_table_num, cudaMemcpyHostToDevice);
//         cudaMemcpy(d_fun_XOR_val, m_functions.m_XOR_val, sizeof(uint32_t) * m_table_num, cudaMemcpyHostToDevice);

//         cudaMemset(d_table, NO_KEY, sizeof(int32_t) * m_table_num * m_capacity);
//     }



// public:
//     Cuckoo_hash(uint32_t table_size, uint32_t function_num): m_capacity_bits(std::floor(std::log2(table_size))), m_capacity(table_size), m_table_num(function_num), m_functions(m_capacity_bits, function_num)
//     {

//         cudaMalloc((void **)&d_table, sizeof(int32_t) * m_table_num * m_capacity);
//         cudaMemset(d_table, 0, sizeof(uint32_t) * m_capacity * m_table_num);
        
//         cudaMalloc((void **)&d_fun_right_shift, sizeof(uint32_t) * m_table_num);
//         cudaMalloc((void **)&d_fun_XOR_val, sizeof(uint32_t) * m_table_num);

//         cudaMemset(d_table, NO_KEY, sizeof(int32_t) * m_table_num * m_capacity);

//         cudaMemcpy(d_fun_right_shift, m_functions.m_right_shift, sizeof(uint32_t) * m_table_num, cudaMemcpyHostToDevice);
//         cudaMemcpy(d_fun_XOR_val, m_functions.m_XOR_val, sizeof(uint32_t) * m_table_num, cudaMemcpyHostToDevice);

//     }

//     ~Cuckoo_hash()
//     {
//         cudaFree(d_table);
//         cudaFree(d_fun_right_shift);
//         cudaFree(d_fun_XOR_val);
//     }

// #ifndef MODIFY
//     bool insert(int32_t *arr, uint32_t len, uint32_t evict_bound = 0)
//     {
//         int32_t *d_arr;
//         cudaMalloc((void**)&d_arr, sizeof(int32_t) * len);
//         cudaMemcpy(d_arr, arr, sizeof(int32_t) * len, cudaMemcpyHostToDevice);

//         bool need_rehash = false;
//         bool *d_need_rehash;
//         cudaMalloc((void**)&d_need_rehash, sizeof(bool));

//         if(evict_bound == 0)
//         {
//             evict_bound = 4 * std::floor(std::log2(len));
//         }
//         uint32_t rehash_times = 0;

//         dim3 block(BLOCK_SIZE);
//         dim3 grid((len + block.x - 1) / block.x);
//         cudaMemset(d_need_rehash, false, sizeof(bool));
//         d_insert_arr<<<grid, block>>>(d_arr, len, d_fun_XOR_val, d_fun_right_shift, m_table_num, m_capacity, d_table, evict_bound, d_need_rehash);
//         cudaDeviceSynchronize();
//         cudaMemcpy(&need_rehash, d_need_rehash, sizeof(bool), cudaMemcpyDeviceToHost);

//         while (need_rehash && rehash_times < rehash_bound)
//         {
//             rehash_times ++;
//             rehash();
//             cudaMemset(d_need_rehash, false, sizeof(bool));
//             d_insert_arr<<<grid, block>>>(d_arr, len, d_fun_XOR_val, d_fun_right_shift, m_table_num, m_capacity, d_table, evict_bound, d_need_rehash);
//             cudaDeviceSynchronize();
//             cudaMemcpy(&need_rehash, d_need_rehash, sizeof(bool), cudaMemcpyDeviceToHost);
//         }
        
//         cudaFree(d_arr);
//         cudaFree(d_need_rehash);
//         std::cout << "rehash times = " << rehash_times << std::endl;

//         return !need_rehash && rehash_times < rehash_bound;
//     }

// #else

//     bool insert(int32_t *arr, uint32_t len, uint32_t evict_bound = 0)
//     {
//         int32_t *d_arr;
//         cudaMalloc((void**)&d_arr, sizeof(int32_t) * len);
//         cudaMemcpy(d_arr, arr, sizeof(int32_t) * len, cudaMemcpyHostToDevice);

//         uint32_t buffer_len = len;
//         uint32_t *d_buffer_len;
//         cudaMalloc((void**)&d_buffer_len, sizeof(uint32_t));
//         int32_t *d_buffer;
//         cudaMalloc((void **)&d_buffer, sizeof(int32_t) * len);

//         if(evict_bound == 0)
//         {
//             evict_bound = 4 * std::floor(std::log2(len));
//         }
//         uint32_t evict_times = 0;
//         uint32_t rehash_times = 0;

//         while(rehash_times < rehash_bound)
//         {
//             do
//             {
//                 cudaMemset(d_buffer_len, 0, sizeof(uint32_t));
//                 dim3 block(BLOCK_SIZE);
//                 dim3 grid((buffer_len + block.x - 1) / block.x);
//                 d_insert_modify<<<grid, block>>>(d_arr, d_buffer, buffer_len, d_buffer_len, d_fun_XOR_val, d_fun_right_shift, m_table_num, m_capacity, d_table);
//                 cudaMemcpy(d_buffer_len, &buffer_len, sizeof(uint32_t), cudaMemcpyDeviceToHost);
//                 std::swap(d_arr, d_buffer);
//                 evict_times ++;
//             } while (buffer_len != 0 && evict_times < evict_bound);

//             if(evict_times == evict_bound)
//             {
//                 rehash_times ++;
//                 rehash();
//             }
//             else
//             {
//                 break;
//             }
//         }

//         return (rehash_times != rehash_bound);        

//     }
// #endif
//     bool* search(int32_t *arr, uint32_t len)
//     {
//         int32_t *d_arr;
//         cudaMalloc((void **)&d_arr, sizeof(int32_t) * len);
//         cudaMemcpy(d_arr, arr, sizeof(int32_t) * len, cudaMemcpyHostToDevice);

//         bool *key_found = (bool *)malloc(sizeof(bool) * len);
//         bool *d_key_found;
//         cudaMalloc((void **)&d_key_found, sizeof(bool) * len);
//         cudaMemset(d_key_found, 0, sizeof(bool) * len);

//         dim3 block(BLOCK_SIZE);
//         dim3 grid((len + block.x - 1) / block.x);

//         d_search_arr<<<grid, block>>>(d_arr, len, d_fun_XOR_val, d_fun_right_shift, m_table_num, m_capacity, d_table, d_key_found);

//         cudaMemcpy(key_found, d_key_found, sizeof(bool) * len, cudaMemcpyDeviceToHost);
//         cudaFree(d_key_found);
//         cudaFree(d_arr);

//         return key_found;
//     }

//     void print_table()
//     {
//         int32_t *table = (int32_t *)malloc(sizeof(int32_t) * m_table_num * m_capacity);
//         cudaMemcpy(table, d_table, sizeof(int32_t) * m_table_num * m_capacity, cudaMemcpyDeviceToHost);

//         for(auto i = 0; i < m_table_num; i ++)
//         {
//             std::cout << "table " << i << " info" << std::endl;
//             std::cout << "XOR_val = " << m_functions.m_XOR_val[i] << " right shift = " << m_functions.m_right_shift[i] << std::endl;
//             for(auto j = 0; j < m_capacity; j ++)
//             {
//                 std::cout << "table index " << j << " value = " << table[i * m_capacity + j] << std::endl;
//             }
//             std::cout << std::endl;
//         }

//         free(table);
//     }
// };



// void Random_array(int32_t *dst, uint32_t size)
// {
//     uint32_t len = size;
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<int32_t> dis(std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
//     for(uint32_t i = 0; i < len; i ++)
//     {
//         dst[i] = dis(gen);
//     }
// }

// void get_Random_arr_from_arr(int32_t *src, int32_t *dst, uint32_t src_len, uint32_t dst_len)
// {
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<int32_t> dis(0, src_len-1);

//     for(auto i = 0; i < dst_len; i ++)
//     {
//         dst[i] = src[dis(gen)];
//     }
// }

void test1()
{
    uint32_t table_size_bit = 25;
    uint32_t table_size = 1 << table_size_bit;
    uint32_t data_start = 10;
    uint32_t data_end = 25;
    uint32_t function_num = 2;
    for(auto data_size_bit = 10 ; data_size_bit < data_end; data_size_bit ++)
    {
        for(auto i = 0; i < REPEAT_TIME; i ++)
        {
            uint32_t data_size = 1 << data_size_bit;
            Cuckoo_hash hash(table_size, function_num);

            int32_t *data = (int32_t *)malloc(sizeof(int32_t) * data_size);
            Random_array(data, data_size);

            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            auto rehash_times = hash.insert(data, data_size) + 1;
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            double duration_time = duration.count();

            std::cout << "time for size_bit " << data_size_bit << " repeat time " << i << "elapse time: " << duration_time <<" speed: " << GET_PERFOEMANCE(data_size * rehash_times, duration_time) << " MOPT/s";
            std::cout << "rehash times : " << rehash_times;
            if(rehash_times != REHASH_BOUND)
            {
                std::cout << " success" << std::endl;
            }
            else
            {
                std::cout << " not success" << std::endl;
            }

            free(data);
        }
        std::cout << "test for data size_bit " << data_size_bit << " end" << std::endl;
        std::cout << std::endl;
    }
}

void test2()
{
    uint32_t table_size_bit = 25;
    uint32_t table_size = 1 << table_size_bit;
    uint32_t function_num = 3;
    uint32_t data_size_bit = 24;
    uint32_t data_size = 1 << data_size_bit;
    uint32_t test_begin = 0;
    uint32_t test_end = 1;
    

    for(auto i = 0; i < 1; i ++)
    {
        Cuckoo_hash hash(table_size, function_num);

        int32_t *data = (int32_t *)malloc(sizeof(int32_t) * data_size);
        Random_array(data, data_size);

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        auto rehash_times = hash.insert(data, data_size);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        double duration_time = duration.count();
        std::cout << "repeat time " << i << " insert speed: " << GET_PERFOEMANCE(data_size, duration_time) << "MOPT/s" << std::endl;
        std::cout << "rehash times : " << rehash_times << std::endl;
        if(rehash_times != REHASH_BOUND)
        {
            std::cout << " success" << std::endl;
        }
        else
        {
            std::cout << " not success" << std::endl;
        }

        for(auto j = test_begin; j < 1; j ++)
        {        
            int percent = 100 - 10 * j;
            uint32_t ori_data_size = data_size * percent / 100;
            uint32_t random_data_size = data_size - ori_data_size;
            int32_t *search_arr = (int32_t *)malloc(sizeof(int32_t) * data_size);

            Random_array(search_arr, random_data_size);
            get_Random_arr_from_arr(data, search_arr+random_data_size, data_size, ori_data_size);
            bool *key_found;

            start = std::chrono::steady_clock::now();
            key_found = hash.search(search_arr, data_size); 
            end = std::chrono::steady_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            duration_time = duration.count();
            std::cout << "data repeat " << percent << "% for search speed: " << GET_PERFOEMANCE(data_size, duration_time) << "MOPT/s" << std::endl;
            // hash.print_table();
            bool found = true;
            uint32_t not_found_num = 0;
            for(auto t = 0UL; t < data_size; t ++)
            {
                if(key_found[t] == false)
                {
                    found = false;
                    // std::cout << "not found index " << t << std::endl;
                    not_found_num ++;
                }
            }
            std::cout << "found " << found << " not found num " << not_found_num << std::endl;
            if(found == false)
            {
                std::cout << "search values" << std::endl;
                // for(auto index = 0; index < data_size; index ++)
                // {
                //     std::cout << "value " << index << " = " << search_arr[index] << std::endl;
                // }
                
            }
            free(key_found);
            free(search_arr);
        }

        free(data);
    }
}

void test3()
{
    std::cout << "----------------------" << std::endl;
    std::cout << "test 3 start" << std::endl << std::endl;
    uint32_t data_size_bit = 24;
    uint32_t data_size = 1 << data_size_bit;

    std::vector<uint32_t> hash_table_size;
    for(auto i = 1; i <= 10; i ++)
    {
        hash_table_size.push_back(data_size * (10 + i) / 10);
    }
    
    hash_table_size.push_back(data_size * (100 + 1) / 100);
    hash_table_size.push_back(data_size * (100 + 2) / 100);
    hash_table_size.push_back(data_size * (100 + 5) / 100);
    uint32_t function_num = 2;

    for(auto i = 0; i < REPEAT_TIME; i ++)
    {
        std::cout << "repeat time " << i << std::endl;
        int32_t *data = (int32_t *)malloc(sizeof(int32_t) * data_size);
        Random_array(data, data_size);

        for(auto j = 0; j < hash_table_size.size(); j ++)
        {
            Cuckoo_hash hash(hash_table_size[j], function_num);

            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            auto success = hash.insert(data, data_size);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            double duration_time = duration.count();

            std::cout << "hash size " << hash_table_size[j] << " speed: " << GET_PERFOEMANCE(data_size, duration_time) << " MOPT/s";
            if(success == true)
            {
                std::cout << " success" << std::endl;
            }
            else
            {
                std::cout << " not success ";
                std::cout << " used time: " << duration_time << std::endl;
            }

        }
        free(data);
    }
}

void test4()
{
    std::cout << "----------------------" << std::endl;
    std::cout << "test 4 start" << std::endl << std::endl;
    uint32_t data_size_bit = 20;
    uint32_t data_size = 1 << data_size_bit;

    uint32_t function_num = 2;
    uint32_t table_size = data_size * 14 / 10;

    std::vector<uint32_t> evict_bounds;
    for(auto i = 1; i <= 10; i ++)
    {
        evict_bounds.push_back(i * data_size_bit);
    }

    for(auto i = 0; i < REPEAT_TIME; i ++)
    {
        Cuckoo_hash(table_size, function_num);
        std::cout << "repeat time " << i << std::endl;
        int32_t *data = (int32_t *)malloc(sizeof(int32_t) * data_size);
        Random_array(data, data_size);

        for(auto j = 0; j < evict_bounds.size(); j ++)
        {
            Cuckoo_hash hash(table_size, function_num);

            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            auto rehash_times = hash.insert(data, data_size, evict_bounds[j]) + 1;
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            double duration_time = duration.count();

            std::cout << "evict bound " << evict_bounds[j] << " speed: " << GET_PERFOEMANCE(data_size * rehash_times, duration_time) << " MOPT/s";
            if(rehash_times == REHASH_BOUND + 1)
            {
                std::cout << " success" << std::endl;
            }
            else
            {
                std::cout << " not success ";
                std::cout << " used time: " << duration_time << std::endl;
            }

        }
    }   
}

int main(int argc, char const *argv[])
{

    test1();
    return 0;
}
