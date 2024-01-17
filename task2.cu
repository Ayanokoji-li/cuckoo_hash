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

#define GET_PERFOEMANCE(data_size, seconds) (data_size / seconds / 1e6)

void test2(uint32_t repeat_time = 5)
{
    uint32_t table_size_bit = 25;
    uint32_t table_size = 1 << table_size_bit;
    uint32_t function_num = 3;
    uint32_t data_size_bit = 24;
    uint32_t data_size = 1 << data_size_bit;
    uint32_t test_begin = 0;
    uint32_t test_end = 11;
    

    for(auto i = 0; i < repeat_time; i ++)
    {
        Cuckoo_hash hash(table_size, function_num);

        int32_t *data = (int32_t *)malloc(sizeof(int32_t) * data_size);
        Random_array(data, data_size);

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        auto rehash_times = hash.insert(data, data_size);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        double duration_time = duration.count();
        std::cout <<"insert speed: " << GET_PERFOEMANCE(data_size * rehash_times, duration_time) << "MOPT/s" << std::endl;

        for(auto j = test_begin; j < test_end; j ++)
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
            std::cout << percent << "% for search time" << duration_time <<"s speed: " << GET_PERFOEMANCE(data_size, duration_time) << "MOPT/s" << std::endl;

            free(key_found);
            free(search_arr);
        }

        free(data);
    }
}

int main(int argc, char const *argv[])
{
    std::string args[] = {"--seed", "--repeat"};
    uint32_t repeat_time = REPEAT_TIME;
    for(auto i = 1; i < argc; i ++)
    {
        if(strcmp(argv[i], args[1].c_str()) == 0)
        {
            repeat_time = atoi(argv[i + 1]);
        }
    }

    test2(repeat_time);

    return 0;
}
