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

void test1(uint32_t data_size_bit, uint32_t repeat_time)
{
    uint32_t table_size_bit = 25;
    uint32_t table_size = 1 << table_size_bit;
    uint32_t function_num = 2;

    for(auto i = 0; i < repeat_time; i ++)
    {
        uint64_t data_size = 1 << data_size_bit;
        Cuckoo_hash hash(table_size, function_num);

        int32_t *data = (int32_t *)malloc(sizeof(int32_t) * data_size);
        Random_array(data, data_size);

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        auto operate_times = hash.insert(data, data_size) + 1;
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        double duration_time = duration.count();

        std::cout << "elapse time: " << duration_time <<"s speed: " << GET_PERFOEMANCE(data_size * operate_times, duration_time) << " MOPT/s" << std::endl;

        free(data);
    }
    std::cout << std::endl;
    
}

int main(int argc, char const *argv[])
{
    std::string args[] = {"--insert", "--repeat"};
    uint32_t repeat_time = REPEAT_TIME;
    uint32_t data_size_bits = 10;

    for(auto i = 1; i < argc; i ++)
    {
        if(strcmp(argv[i], args[0].c_str()) == 0)
        {
            data_size_bits = atoi(argv[i + 1]);
        }
        else if(strcmp(argv[i], args[1].c_str()) == 0)
        {
            repeat_time = atoi(argv[i + 1]);
        }
    }
    test1(data_size_bits, repeat_time);

    return 0;
}
