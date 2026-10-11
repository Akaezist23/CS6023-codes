// Graph generator for the A3 delta-stepping testbench.
// Usage: gen <out> <N> <E> <S> <delta> <K> <type> <wmin> <wmax> <seed>
//   type: rand  - uniform random edges
//         hub   - ~30% of edges leave 8 hub vertices, ~10% of targets are hubs (skewed degrees)
//         chain - path 0->1->...->N-1 plus short forward edges (long diameter, many buckets)
//         sparse- many vertices with no out-edges, many unreachable ones
// Edges are produced directly in CSR order, so memory is O(N + E) with small constants.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <string>

static FILE* fout;
static char buf[1 << 22];
static size_t bpos = 0;

static inline void flush_buf() { fwrite(buf, 1, bpos, fout); bpos = 0; }
static inline void put_int(long long x, char sep) {
    if (bpos + 24 > sizeof(buf)) flush_buf();
    if (x < 0) { buf[bpos++] = '-'; x = -x; }
    char tmp[24]; int n = 0;
    do { tmp[n++] = '0' + x % 10; x /= 10; } while (x);
    while (n) buf[bpos++] = tmp[--n];
    buf[bpos++] = sep;
}

int main(int argc, char** argv) {
    if (argc < 11) {
        fprintf(stderr, "usage: gen out N E S delta K type wmin wmax seed\n");
        return 1;
    }
    const char* out = argv[1];
    int N = atoi(argv[2]);
    long long E = atoll(argv[3]);
    int S = atoi(argv[4]);
    int delta = atoi(argv[5]);
    int K = atoi(argv[6]);
    std::string type = argv[7];
    int wmin = atoi(argv[8]), wmax = atoi(argv[9]);
    uint64_t seed = strtoull(argv[10], nullptr, 10);

    std::mt19937_64 rng(seed);
    auto rnd = [&](uint64_t n) { return (uint64_t)(rng() % n); };

    // ---- degrees ----
    std::vector<long long> deg(N, 0);
    const int HUBS = std::min(8, N);
    if (type == "rand") {
        for (long long i = 0; i < E; i++) deg[rnd(N)]++;
    } else if (type == "hub") {
        long long hubE = E * 3 / 10;
        for (long long i = 0; i < hubE; i++) deg[rnd(HUBS)]++;
        for (long long i = hubE; i < E; i++) deg[rnd(N)]++;
    } else if (type == "chain") {
        long long chainE = std::min<long long>(E, N - 1);
        for (long long u = 0; u < chainE; u++) deg[u]++;
        for (long long i = chainE; i < E; i++) deg[rnd(N)]++;
    } else if (type == "sparse") {
        // only ~1/4 of the vertices have out-edges
        int active = std::max(1, N / 4);
        for (long long i = 0; i < E; i++) deg[rnd(active) * 4 % N]++;
    } else {
        fprintf(stderr, "unknown type %s\n", type.c_str());
        return 1;
    }

    // ---- edges in CSR order ----
    std::vector<int> neighs(E);
    std::vector<uint16_t> weights(E);
    long long idx = 0;
    for (int u = 0; u < N; u++) {
        for (long long j = 0; j < deg[u]; j++) {
            int v;
            if (type == "hub") {
                v = (rnd(10) == 0) ? (int)rnd(HUBS) : (int)rnd(N);
            } else if (type == "chain") {
                if (j == 0 && u < N - 1) v = u + 1;
                else v = (int)((u + 1 + rnd(64)) % N);
            } else if (type == "sparse") {
                v = (int)rnd(N);
                if (v % 2 == 1) v = (v + 1) % N;   // odd vertices mostly unreachable
            } else {
                v = (int)rnd(N);
            }
            neighs[idx] = v;
            weights[idx] = (uint16_t)(wmin + rnd(wmax - wmin + 1));
            idx++;
        }
    }

    // ---- sources: vertex 0 first (interesting for chain), the rest random ----
    std::vector<int> src(S);
    for (int i = 0; i < S; i++) src[i] = (i == 0) ? 0 : (int)rnd(N);
    if (type == "hub" && S > 1) src[1] = 1 % N;  // a hub as source
    if (type == "hub" && S > 2) src[2] = N - 1;  // a likely ordinary vertex

    fout = fopen(out, "wb");
    if (!fout) { perror(out); return 1; }
    put_int(delta, ' '); put_int(K, '\n');
    put_int(N, ' '); put_int(E, ' '); put_int(S, '\n');
    for (int i = 0; i < S; i++) put_int(src[i], i + 1 == S ? '\n' : ' ');
    long long off = 0;
    for (int u = 0; u <= N; u++) {
        put_int(off, u == N ? '\n' : ' ');
        if (u < N) off += deg[u];
    }
    for (long long i = 0; i < E; i++) put_int(neighs[i], i + 1 == E ? '\n' : ' ');
    if (E == 0) buf[bpos++] = '\n';
    for (long long i = 0; i < E; i++) put_int(weights[i], i + 1 == E ? '\n' : ' ');
    flush_buf();
    fclose(fout);
    return 0;
}
