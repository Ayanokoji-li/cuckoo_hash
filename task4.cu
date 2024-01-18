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

void test4(double ratio, uint32_t repeat_time = REPEAT_TIME, uint32_t function_num = 3, const std::string& file_name = "", bool from_file = false)
{
    uint32_t data_size_bit = 24;
    uint32_t data_size = 1 << data_size_bit;

    uint32_t table_size = data_size * 14 / 10;

    std::vector<uint32_t> evict_bounds;
    evict_bounds.push_back(std::floor(data_size_bit * ratio));
    std::cout << "evict bound: " << evict_bounds[0] << std::endl;
    for(auto i = 0; i < repeat_time; i ++)
    {
        Cuckoo_hash(table_size, function_num);
        int32_t *data = (int32_t *)malloc(sizeof(int32_t) * data_size);
        // Random_array(data, data_size);
        get_array(data, data_size, from_file, file_name);

        Cuckoo_hash hash(table_size, function_num);

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        auto rehash_times = hash.insert(data, data_size, evict_bounds[0]) + 1;
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        double duration_time = duration.count();

        std::cout << "time: "<< duration_time << "s speed: " << GET_PERFOEMANCE(data_size * rehash_times, duration_time) << " MOPT/s" << std::endl;
    }   
    std::cout << std::endl;
}

int main(int argc, char const *argv[])
{
    std::string args[] = {"--ratio", "--repeat", "--fucntion_num", "--file-in"};
    uint32_t repeat_time = REPEAT_TIME;
    double ratio = 0.2;
    uint32_t function_num = 2;
    bool from_file = false;
    std::string file_name;

    for(auto i = 1; i < argc; i ++)
    {
        if(strcmp(argv[i], args[0].c_str()) == 0)
        {
            ratio = atof(argv[i + 1]);
        }
        else if(strcmp(argv[i], args[1].c_str()) == 0)
        {
            repeat_time = atoi(argv[i + 1]);
        }
        else if(strcmp(argv[i], args[2].c_str()) == 0)
        {
            function_num = atoi(argv[i + 1]);
        }
        else if(strcmp(argv[i], args[3].c_str()) == 0)
        {
            from_file = true;
            file_name = argv[i + 1];
        }
    }

    test4(ratio, repeat_time, function_num, file_name, from_file);

    return 0;
}
