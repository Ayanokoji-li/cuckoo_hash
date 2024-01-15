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

class Hash_function
{
private:
    uint32_t XOR_val;
    uint32_t right_shift;
    uint32_t capacity_bits;

public:
    Hash_function(uint32_t capacity_bits) : capacity_bits(capacity_bits)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0, (uint32_t)(-1));
        XOR_val = dis(gen);
        std::uniform_int_distribution<uint32_t> dis2(0, 32 - capacity_bits);
        right_shift = dis2(gen);
    }

    uint32_t operator==(const Hash_function& other)
    {
        return (XOR_val == other.XOR_val) && (right_shift == other.right_shift) && (capacity_bits == other.capacity_bits);
    }

    uint32_t operator()(uint32_t key)
    {
        return (((key * key) ^ XOR_val) >> right_shift) % (1 << capacity_bits);
    }
};

void chech_and_rebuild(Hash_function *function_arr, uint32_t function_num, uint32_t capacity_bits)
{
    for(uint64_t i = 1; i < function_num; i ++)
    {
        for(uint64_t j = 0; j < i ; j ++)
        {
            if(function_arr[i] == function_arr[j])
            {
                function_arr[j] = Hash_function(capacity_bits);
            }
        }
    }
}

class Cuckoo_hash
{
private:
    uint32_t capacity_bits;
    uint32_t capacity;
    uint32_t table_num;
    uint32_t **table;
    Hash_function *functions;

public:
    Cuckoo_hash(uint32_t table_size_bit, uint32_t function_num): capacity_bits(table_size_bit), capacity(1 << table_size_bit), table_num(function_num)
    {
        functions = (Hash_function*)malloc(sizeof(Hash_function) * function_num);
        table = (uint32_t **)malloc(sizeof(uint32_t *) * table_num);
        for(auto i = 0; i < function_num; i ++)
        {
            functions[i] = Hash_function(capacity_bits);
            table[i] = (uint32_t *)malloc(sizeof(uint32_t) * capacity);
            memset(table[i], 0, sizeof(uint32_t) * capacity);
        }
        chech_and_rebuild(functions, function_num, capacity_bits);
        
    }

    ~Cuckoo_hash()
    {
        free(functions);
        for(auto i = 0; i < table_num; i ++)
        {
            free(table[i]);
        }
        free(table);
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
    Hash_function test(25);
    uint32_t random_size_bits = 3;
    int32_t *random_arr = (int32_t *)malloc(sizeof(int32_t) * (1 << random_size_bits));
    Random_array(random_arr, random_size_bits);
    for(auto i = 0UL; i < (1 << 3) ; i ++)
    {
        std::cout << "hash res :" << test(random_arr[i]) << std::endl;
    }
    std::cout << sizeof(Hash_function) << std::endl;
    return 0;
}
