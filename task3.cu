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

void test3(double ratio, uint32_t repeat_time)
{
    uint32_t data_size_bit = 24;
    uint32_t data_size = 1 << data_size_bit;
    uint32_t function_num = 3;

    std::vector<uint32_t> hash_table_size;
    hash_table_size.push_back(std::floor(ratio * data_size));
    std::cout << "hash size " << hash_table_size[0] << std::endl;

    for(auto i = 0; i < REPEAT_TIME; i ++)
    {
        int32_t *data = (int32_t *)malloc(sizeof(int32_t) * data_size);
        Random_array(data, data_size);

        for(auto j = 0; j < hash_table_size.size(); j ++)
        {
            Cuckoo_hash hash(hash_table_size[j], function_num);

            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            auto repeat_time = hash.insert(data, data_size) + 1;
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            double duration_time = duration.count();

            if(repeat_time != REHASH_BOUND + 1)
            {
                std::cout << " success" ;
            }
            else
            {
                std::cout << " not success ";
            }
            std::cout  << " times "<< duration_time <<"s speed: " << GET_PERFOEMANCE(data_size * repeat_time, duration_time) << " MOPT/s" << std::endl;

        }
        free(data);
    }
}

int main(int argc, char const *argv[])
{
    std::string args[] = {"--ratio", "--repeat"};
    uint32_t repeat_time = REPEAT_TIME;
    double ratio = 1.4;
    for(auto i = 1; i < argc; i ++)
    {
        if(strcmp(argv[i], args[0].c_str()) == 0)
        {
            ratio = atof(argv[i + 1]);
            std::cout << "args " << ratio << std::endl;
        }
        else if(strcmp(argv[i], args[1].c_str()) == 0)
        {
            repeat_time = atoi(argv[i + 1]);
        }
    }

    test3(ratio, repeat_time);
    return 0;
}
