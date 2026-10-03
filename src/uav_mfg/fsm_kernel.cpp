#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <omp.h>

// Neighbors of a seven-point stencil always occupy different i+j+k planes.
// Every point on one plane can therefore be updated concurrently, while the
// implicit OpenMP barrier preserves the directional Gauss-Seidel dependencies.
extern "C" int fsm_solve(double* phi, const double* speed, const uint8_t* active,
                         const uint8_t* target, int nx, int ny, int nz,
                         const double* spacing, int cycles, double tolerance,
                         int threads, double* maximum_update) {
    const int64_t size = int64_t(nx) * ny * nz;
    const double infinity = std::numeric_limits<double>::infinity();
    for (int64_t p = 0; p < size; ++p) phi[p] = target[p] ? 0.0 : infinity;
    std::vector<double> previous(size);
    for (int cycle = 0; cycle < cycles; ++cycle) {
        std::copy(phi, phi + size, previous.begin());
        #pragma omp parallel num_threads(threads)
        {
            for (int signs = 0; signs < 8; ++signs) {
                for (int plane = 0; plane <= nx + ny + nz - 3; ++plane) {
                    #pragma omp for schedule(static)
                    for (int a = 0; a < nx; ++a) {
                        const int i = (signs & 1) ? nx - 1 - a : a;
                        const int first = std::max(0, plane - a - nz + 1);
                        const int last = std::min(ny - 1, plane - a);
                        for (int b = first; b <= last; ++b) {
                            const int c = plane - a - b;
                            const int j = (signs & 2) ? ny - 1 - b : b;
                            const int k = (signs & 4) ? nz - 1 - c : c;
                            const int64_t p = (int64_t(i) * ny + j) * nz + k;
                            if (!active[p]) continue;
                            double neighbors[3] = {
                                std::min(i > 0 ? phi[p - ny * nz] : infinity,
                                         i + 1 < nx ? phi[p + ny * nz] : infinity),
                                std::min(j > 0 ? phi[p - nz] : infinity,
                                         j + 1 < ny ? phi[p + nz] : infinity),
                                std::min(k > 0 ? phi[p - 1] : infinity,
                                         k + 1 < nz ? phi[p + 1] : infinity)};
                            int order[3] = {0, 1, 2};
                            std::sort(order, order + 3, [&](int x, int y) { return neighbors[x] < neighbors[y]; });
                            double aa = 0, bb = 0, cc = -1.0 / (speed[p] * speed[p]);
                            for (int count = 0; count < 3; ++count) {
                                const int d = order[count];
                                if (!std::isfinite(neighbors[d])) break;
                                const double weight = 1.0 / (spacing[d] * spacing[d]);
                                aa += weight; bb -= 2 * neighbors[d] * weight;
                                cc += neighbors[d] * neighbors[d] * weight;
                                const double candidate = (-bb + std::sqrt(std::max(0.0, bb * bb - 4 * aa * cc))) / (2 * aa);
                                if (count == 2 || candidate <= neighbors[order[count + 1]]) {
                                    phi[p] = std::min(phi[p], candidate);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        double change = 0;
        #pragma omp parallel for num_threads(threads) reduction(max:change)
        for (int64_t p = 0; p < size; ++p) {
            if (active[p]) {
                const double delta = std::isfinite(phi[p]) && std::isfinite(previous[p])
                    ? std::abs(phi[p] - previous[p]) : infinity;
                change = std::max(change, delta);
            }
        }
        *maximum_update = change;
        if (change < tolerance) return cycle + 1;
    }
    for (int64_t p = 0; p < size; ++p) if (active[p] && !std::isfinite(phi[p])) return -1;
    return 0;
}
