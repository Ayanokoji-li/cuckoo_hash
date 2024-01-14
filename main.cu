#include <iostream>
#include <cuda.h>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <limits>

template <uint32_t capacity_bits>
class Hash_function
{
private:
    uint32_t XOR_val;
    uint32_t right_shift;

public:
    Hash_function()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0, std::numeric_limits::uint32_t());
        XOR_val = dis(gen);
        right_shift = dis(gen);
    }
}