#include "SmokeSim.h"
#include "CudaArray.cuh"
#include "helper_math.h"

#ifndef N
#define N 128
#endif

template <class T>
__global__ void fill_zero_kernel(CudaSurfaceAccessor<T> suracc)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= 128 || y >= 128 || z >= 128)
        return;

    suracc.write(T{0}, x, y, z);
}

__global__ void change_velocity_density_kernel(CudaSurfaceAccessor<float> density, CudaSurfaceAccessor<float4> velocity, CudaSurfaceAccessor<float> bound)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= 128 || y >= 128 || z >= 128)
        return;
    if (bound.read(x, y, z) > 0)
    {
        density.write(1, x, y, z);
        velocity.write(make_float4(0, 0, 0.5f, 0), x, y, z);
    }
}

__global__ void advect_kernel(CudaSurfaceAccessor<float4> nextPos, CudaTextureAccessor<float4> velocity, CudaTextureAccessor<float> bound)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= 128 || y >= 128 || z >= 128)
        return;

    auto sample = [](CudaTextureAccessor<float4> velAcc, float3 pos, float3 vel, float time) -> float3
    {
        pos = pos - vel * time;
        float4 res = velAcc.sample(pos.x, pos.y, pos.z);
        return make_float3(res.x, res.y, res.z);
    };

    float3 pos = make_float3(x + 0.5f, y + 0.5f, z + 0.5f);

    if (bound.sample(x, y, z) < 0)
    {
        float3 k1 = sample(velocity, pos, make_float3(0, 0, 0), 0);
        float3 k2 = sample(velocity, pos, k1, 0.5f);
        float3 k3 = sample(velocity, pos, k2, 0.5f);
        float3 k4 = sample(velocity, pos, k3, 1);

        pos -= (k1 + 2 * k2 + 2 * k3 + k4) / 6.0f;
    }
    nextPos.write(make_float4(pos.x, pos.y, pos.z, 0), x, y, z);
}

template <class T>
__global__ void sample_kernel(CudaSurfaceAccessor<T> sufAcc, CudaTextureAccessor<T> texAcc, CudaSurfaceAccessor<float4> nextPos)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= 128 || y >= 128 || z >= 128)
        return;

    float4 next_pos = nextPos.read(x, y, z);
    T res = texAcc.sample(next_pos.x, next_pos.y, next_pos.z);
    sufAcc.write(res, x, y, z);
}

__global__ void cal_div_v(CudaSurfaceAccessor<float> div_v, CudaSurfaceAccessor<float4> v)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= 128 || y >= 128 || z >= 128)
        return;

    float vr = v.read<cudaBoundaryModeClamp>(x + 1, y, z).x;
    float vl = v.read<cudaBoundaryModeClamp>(x - 1, y, z).x;
    float vu = v.read<cudaBoundaryModeClamp>(x, y + 1, z).y;
    float vd = v.read<cudaBoundaryModeClamp>(x, y - 1, z).y;
    float vf = v.read<cudaBoundaryModeClamp>(x, y, z + 1).z;
    float vb = v.read<cudaBoundaryModeClamp>(x, y, z - 1).z;
    div_v.write((vr - vl + vu - vd + vf - vb) * 0.5f, x, y, z);
}

__global__ void jacobi_kernel(CudaSurfaceAccessor<float> pressureNext, CudaSurfaceAccessor<float> pressure, CudaSurfaceAccessor<float> div_v)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= 128 || y >= 128 || z >= 128)
        return;
    float r = pressure.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float l = pressure.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float u = pressure.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float d = pressure.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float f = pressure.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float b = pressure.read<cudaBoundaryModeClamp>(x, y, z - 1);

    float div = div_v.read<cudaBoundaryModeClamp>(x, y, z);
    float res = (r + l + u + d + f + b - div) / 6.0f;
    pressureNext.write(res, x, y, z);
}

__global__ void projection_kernel(CudaSurfaceAccessor<float4> v, CudaSurfaceAccessor<float> pressure)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= 128 || y >= 128 || z >= 128)
        return;

    float r = pressure.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float l = pressure.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float u = pressure.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float d = pressure.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float f = pressure.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float b = pressure.read<cudaBoundaryModeClamp>(x, y, z - 1);

    float4 vel = v.read(x, y, z);
    vel.x -= (r - l) * 0.5f;
    vel.y -= (u - d) * 0.5f;
    vel.z -= (f - b) * 0.5f;
    v.write(vel, x, y, z);
}

struct SmokeSim::SmokeSimImpl
{
    std::unique_ptr<CudaTexture<float>> density;
    std::unique_ptr<CudaTexture<float4>> velocity;
    std::unique_ptr<CudaTexture<float>> bound;
    std::unique_ptr<CudaSurface<float4>> nextPos;
    std::unique_ptr<CudaSurface<float>> div_v;
    std::unique_ptr<CudaTexture<float>> pressure;

    std::unique_ptr<CudaTexture<float>> float_buff;
    std::unique_ptr<CudaTexture<float4>> float4_buff;

    uint3 dim;
};

SmokeSim::SmokeSim(unsigned int x, unsigned int y, unsigned int z)
{
    Impl = std::make_unique<SmokeSimImpl>();
    Impl->density = std::make_unique<CudaTexture<float>>(uint3{x, y, z});
    Impl->velocity = std::make_unique<CudaTexture<float4>>(uint3{x, y, z});
    Impl->bound = std::make_unique<CudaTexture<float>>(uint3{x, y, z});
    Impl->nextPos = std::make_unique<CudaSurface<float4>>(uint3{x, y, z});
    Impl->div_v = std::make_unique<CudaSurface<float>>(uint3{x, y, z});
    Impl->pressure = std::make_unique<CudaTexture<float>>(uint3{x, y, z});

    Impl->float_buff = std::make_unique<CudaTexture<float>>(uint3{x, y, z});
    Impl->float4_buff = std::make_unique<CudaTexture<float4>>(uint3{x, y, z});

    Impl->dim = uint3{x, y, z};

    fill_zero_kernel<<<{(N + 7) / 8, (N + 7) / 8, (N + 7) / 8}, {8, 8, 8}>>>(Impl->pressure->accessSurface());
}

SmokeSim::~SmokeSim()
{
}

void SmokeSim::copyInDensity(float *data)
{
    Impl->density->copyIn(data);
}

void SmokeSim::copyInVelocity(float *data)
{
    Impl->velocity->copyIn((float4 *)data);
}

void SmokeSim::copyInBound(float *data)
{
    Impl->bound->copyIn(data);
}

void SmokeSim::copyOutDensity(float *data)
{
    Impl->density->copyOut(data);
}

void SmokeSim::advect()
{
    fill_zero_kernel<<<{(N + 7) / 8, (N + 7) / 8, (N + 7) / 8}, {8, 8, 8}>>>(
        Impl->pressure->accessSurface());

    change_velocity_density_kernel<<<{(N + 7) / 8, (N + 7) / 8, (N + 7) / 8}, {8, 8, 8}>>>(
        Impl->density->accessSurface(),Impl->velocity->accessSurface(),Impl->bound->accessSurface());

    advect_kernel<<<{(N + 7) / 8, (N + 7) / 8, (N + 7) / 8}, {8, 8, 8}>>>(
        Impl->nextPos->accessSurface(), Impl->velocity->accessTexture(), Impl->bound->accessTexture());

    // sample for density
    sample_kernel<<<{(N + 7) / 8, (N + 7) / 8, (N + 7) / 8}, {8, 8, 8}>>>(
        Impl->float_buff->accessSurface(), Impl->density->accessTexture(), Impl->nextPos->accessSurface());

    std::swap(Impl->float_buff, Impl->density);

    // sample for velocity
    sample_kernel<<<{(N + 7) / 8, (N + 7) / 8, (N + 7) / 8}, {8, 8, 8}>>>(
        Impl->float4_buff->accessSurface(), Impl->velocity->accessTexture(), Impl->nextPos->accessSurface());

    std::swap(Impl->float4_buff, Impl->velocity);
}

void SmokeSim::projection()
{
    cal_div_v<<<{(N + 7) / 8, (N + 7) / 8, (N + 7) / 8}, {8, 8, 8}>>>(
        Impl->div_v->accessSurface(), Impl->velocity->accessSurface());

    int times = 50;
    for (int i = 0; i < times; i++)
    {
        jacobi_kernel<<<{(N + 7) / 8, (N + 7) / 8, (N + 7) / 8}, {8, 8, 8}>>>(
            Impl->float_buff->accessSurface(), Impl->pressure->accessSurface(),
            Impl->div_v->accessSurface());
        std::swap(Impl->float_buff, Impl->pressure);
    }

    projection_kernel<<<{(N + 7) / 8, (N + 7) / 8, (N + 7) / 8}, {8, 8, 8}>>>(
        Impl->velocity->accessSurface(), Impl->pressure->accessSurface());
}

void SmokeSim::step()
{
    for (int i = 0; i < 24; i++)
    {
        advect();
        projection();
    }
}
