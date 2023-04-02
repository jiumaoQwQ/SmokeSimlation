#pragma once

#include <memory>

struct SmokeSim
{
private:
    struct SmokeSimImpl;
    std::unique_ptr<SmokeSimImpl> Impl;

    void advect();
    void projection();

public:
    explicit SmokeSim(unsigned int x, unsigned int y, unsigned int z);
    void copyInDensity(float *data);
    void copyInVelocity(float *data);
    void copyInBound(float *data);
    void copyOutDensity(float *data);
    void step();
    ~SmokeSim();
};