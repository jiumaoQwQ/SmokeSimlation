#include <vector>
#include <cmath>
#include <iostream>

#include "vdbhelper.h"
#include "SmokeSim.h"

#define N 128

struct int3
{
    int x, y, z;
};

int main()
{
    int3 dim{128, 128, 128};
    SmokeSim smokesim(N, N, N);
    int3 sphere_pos = {128 / 2, 128 / 2, 128 / 2 - 30};
    float r = 16;
    {
        std::vector<float> cpu_data(dim.x * dim.y * dim.z);

        for (int z = 0; z < dim.z; z++)
        {
            for (int y = 0; y < dim.y; y++)
            {
                for (int x = 0; x < dim.x; x++)
                {
                    float res = 0;
                    if (std::hypot(x - sphere_pos.x, y - sphere_pos.y, z - sphere_pos.z) <= r)
                    {
                        res = 1;
                    }
                    else
                    {
                        res = -1;
                    }
                    cpu_data[x + y * dim.x + z * dim.x * dim.y] = res;
                }
            }
        }
        smokesim.copyInBound(cpu_data.data());
    }

    {
        std::vector<float> cpu_data(dim.x * dim.y * dim.z);

        for (int z = 0; z < dim.z; z++)
        {
            for (int y = 0; y < dim.y; y++)
            {
                for (int x = 0; x < dim.x; x++)
                {
                    if (std::hypot(x - sphere_pos.x, y - sphere_pos.y, z - sphere_pos.z) <= r)
                        cpu_data[x + y * dim.x + z * dim.x * dim.y] = 1;
                }
            }
        }
        smokesim.copyInDensity(cpu_data.data());
    }
    {
        std::vector<float> cpu_data(dim.x * dim.y * dim.z);

        for (int z = 0; z < dim.z; z++)
        {
            for (int y = 0; y < dim.y; y++)
            {
                for (int x = 0; x < dim.x; x++)
                {
                    if (std::hypot(x - sphere_pos.x, y - sphere_pos.y, z - sphere_pos.z) <= r)
                        cpu_data[x + y * dim.x + z * dim.x * dim.y] = 1;
                }
            }
        }
        smokesim.copyInDensity(cpu_data.data());
    }
    {
        std::vector<float> cpu_data(4 * dim.x * dim.y * dim.z);

        for (int z = 0; z < dim.z; z++)
        {
            for (int y = 0; y < dim.y; y++)
            {
                for (int x = 0; x < dim.x; x++)
                {
                    if (std::hypot(x - sphere_pos.x, y - sphere_pos.y, z - sphere_pos.z) <= r)
                    {
                        cpu_data[4 * (x + y * dim.x + z * dim.x * dim.y) + 0] = 0;
                        cpu_data[4 * (x + y * dim.x + z * dim.x * dim.y) + 1] = 0;
                        cpu_data[4 * (x + y * dim.x + z * dim.x * dim.y) + 2] = 0.5f;
                        cpu_data[4 * (x + y * dim.x + z * dim.x * dim.y) + 3] = 0;
                    }
                }
            }
        }
        smokesim.copyInVelocity(cpu_data.data());
    }
    std::vector<float> cpu_density(dim.x * dim.y * dim.z);

    for (int i = 0; i < 120; i++)
    {
        smokesim.step();
        smokesim.copyOutDensity(cpu_density.data());
        writeDense("dense" + std::to_string(1000 + i).substr(1) + ".vdb", cpu_density.data(), dim.x, dim.y, dim.z);
    }
    return 0;
}