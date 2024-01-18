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

void test1(uint32_t data_size_bit, uint32_t repeat_time, uint32_t function_num, bool from_file = false, const std::string &file_name = "")
{
    uint32_t table_size_bit = 25;
    uint32_t table_size = 1 << table_size_bit;
    std::vector<double> times;
    std::cout << "data size bits " << data_size_bit << std::endl;
    for(auto i = 0; i < repeat_time; i ++)
    {
        uint32_t data_size = 1 << data_size_bit;
        Cuckoo_hash hash(table_size, function_num);

        int32_t *data = (int32_t *)malloc(sizeof(int32_t) * data_size);
        get_array(data, data_size, from_file, file_name);
        // Random_array(data, data_size);

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        auto operate_times = hash.insert(data, data_size) + 1;
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        double duration_time = duration.count();

        std::cout << "elapse time: " << duration_time <<"s speed: " << GET_PERFOEMANCE(data_size * operate_times, duration_time) << " MOPT/s" << std::endl;
        times.push_back(duration_time);
        free(data);
    }

    double average = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << "average time: " << average << "s" << std::endl;

    std::cout << std::endl;
    
}

int main(int argc, char const *argv[])
{
    std::string args[] = {"--insert", "--repeat", "--function_num", "--file-in"};
    uint32_t repeat_time = 100;
    uint32_t data_size_bits = 24;
    uint32_t function_num = 2;
    bool file_in = false;
    std::string file_name;
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
        else if(strcmp(argv[i], args[2].c_str()) == 0)
        {
            function_num = atoi(argv[i + 1]);
        }
        else if(strcmp(argv[i], args[3].c_str()) == 0)
        {
            file_in = true;
            file_name = argv[i + 1];
        }
    }
    test1(data_size_bits, repeat_time, function_num, file_in, file_name);

    return 0;
}
