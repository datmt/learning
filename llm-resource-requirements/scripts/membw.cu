// Memory-bandwidth microbenchmark for the GPU side of a unified-memory system.
// Measures sustained read, write and copy bandwidth with simple vectorised (16-byte) streaming kernels.
//
// Build: nvcc -O3 -arch=sm_121 -o membw membw.cu
// Run:   ./membw [iters]
//
// Limitations (also stated in the post):
//  * This is a *simple streaming kernel*, not the access pattern of llama.cpp's matmul kernels.
//    It estimates what a well-behaved GPU kernel can pull from memory; it is not a hardware peak.
//  * Bandwidth is computed as bytes / time from CUDA events. Bytes are what the kernel requests,
//    not hardware counters.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t ck_err_ = (x); if (ck_err_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(ck_err_)); exit(1);} } while (0)

__global__ void k_read(const uint4* __restrict__ a, size_t n, unsigned* out) {
    unsigned acc = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint4 v = a[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;   // keep the loads alive
    }
    if (acc == 0x9e3779b9u) out[blockIdx.x] = acc;  // practically never true, prevents dead-code elimination
}

__global__ void k_write(uint4* __restrict__ a, size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 v = make_uint4(1, 2, 3, 4);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) a[i] = v;
}

__global__ void k_copy(const uint4* __restrict__ a, uint4* __restrict__ b, size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) b[i] = a[i];
}

template <typename F>
static std::vector<float> timeit(F f, int warm, int iters) {
    cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    for (int i = 0; i < warm; i++) f();
    CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < iters; i++) {
        CK(cudaEventRecord(s)); f(); CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
        float t; CK(cudaEventElapsedTime(&t, s, e)); ms.push_back(t);
    }
    return ms;
}

static void report(const char* name, size_t bytes_moved, size_t buf_bytes, std::vector<float> ms) {
    std::sort(ms.begin(), ms.end());
    double med = ms[ms.size() / 2];
    auto gbs = [&](double t) { return bytes_moved / (t / 1e3) / 1e9; };
    // JSON line: easy to parse; GB/s = 1e9 bytes/s, GiB/s = 2^30 bytes/s
    printf("{\"test\":\"%s\",\"buffer_GiB\":%.2f,\"iters\":%zu,"
           "\"GBps_median\":%.1f,\"GBps_max\":%.1f,\"GBps_min\":%.1f,"
           "\"GiBps_median\":%.1f}\n",
           name, buf_bytes / 1073741824.0, ms.size(), gbs(med), gbs(ms.front()), gbs(ms.back()),
           bytes_moved / (med / 1e3) / 1073741824.0);
    fflush(stdout);
}

int main(int argc, char** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 20;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    size_t freeB, totalB; CK(cudaMemGetInfo(&freeB, &totalB));
    fprintf(stderr, "device=%s sm=%d.%d SMs=%d L2=%.0f MiB free=%.1f GiB total=%.1f GiB\n", p.name, p.major, p.minor,
            p.multiProcessorCount, p.l2CacheSize / 1048576.0, freeB / 1073741824.0, totalB / 1073741824.0);
    int blocks = p.multiProcessorCount * 8, threads = 256;
    unsigned* sink; CK(cudaMalloc(&sink, blocks * sizeof(unsigned)));

    const double sizes_gib[] = {0.0625, 1, 4, 8, 16};
    for (double g : sizes_gib) {
        size_t bytes = (size_t)(g * 1073741824.0);
        size_t n = bytes / sizeof(uint4);
        // cudaMemGetInfo on a unified-memory system does not count reclaimable page cache as free,
        // so do not trust it as a limit: just try the allocation. Large sizes use a single buffer (no copy test).
        bool two = g <= 4;
        uint4 *a = nullptr, *b = nullptr;
        if (cudaMalloc(&a, bytes) != cudaSuccess || (two && cudaMalloc(&b, bytes) != cudaSuccess)) {
            fprintf(stderr, "skip %.2f GiB (allocation failed)\n", g); cudaGetLastError(); continue;
        }
        CK(cudaMemset(a, 1, bytes)); if (two) CK(cudaMemset(b, 2, bytes));
        CK(cudaDeviceSynchronize());
        report("read",  bytes,     bytes, timeit([&] { k_read<<<blocks, threads>>>(a, n, sink); }, 3, iters));
        report("write", bytes,     bytes, timeit([&] { k_write<<<blocks, threads>>>(a, n); }, 3, iters));
        if (two) report("copy",  bytes * 2, bytes, timeit([&] { k_copy<<<blocks, threads>>>(a, b, n); }, 3, iters));
        CK(cudaFree(a)); if (two) CK(cudaFree(b));
    }
    return 0;
}
