// CPU reference for the A3 testbench.
// Usage: ref <input> <output>
// Distances come from Dijkstra. In adaptive mode (delta == -1) the delta log is
// reproduced from the spec (section 11): after each bucket, every vertex with
// dist <= cutoff is settled, the Far keys are the best tentative distances through
// settled vertices, and
//   Dmin = min key, cutoff = key[min(K-1, |far|-1)], log cutoff - Dmin.
// The first batch (cutoff = 0) is not logged.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <queue>
#include <set>
#include <algorithm>
#include <functional>

static const int INF = 2147483647;

static std::vector<char> data;
static size_t pos = 0;
static long long read_ll() {
    while (pos < data.size() && (data[pos] == ' ' || data[pos] == '\n' || data[pos] == '\r' || data[pos] == '\t')) pos++;
    bool neg = false;
    if (data[pos] == '-') { neg = true; pos++; }
    long long x = 0;
    while (pos < data.size() && data[pos] >= '0' && data[pos] <= '9') x = x * 10 + (data[pos++] - '0');
    return neg ? -x : x;
}

static char obuf[1 << 22];
static size_t opos = 0;
static FILE* fout;
static void put_int(long long x) {
    if (opos + 24 > sizeof(obuf)) { fwrite(obuf, 1, opos, fout); opos = 0; }
    if (x < 0) { obuf[opos++] = '-'; x = -x; }
    char t[24]; int n = 0;
    do { t[n++] = '0' + x % 10; x /= 10; } while (x);
    while (n) obuf[opos++] = t[--n];
    obuf[opos++] = '\n';
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: ref in out\n"); return 1; }
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 1; }
    fseek(fin, 0, SEEK_END); long long sz = ftell(fin); fseek(fin, 0, SEEK_SET);
    data.resize(sz + 1);
    if (fread(data.data(), 1, sz, fin) != (size_t)sz) { fprintf(stderr, "read error\n"); return 1; }
    data[sz] = '\n';
    fclose(fin);

    int delta = (int)read_ll(), K = (int)read_ll();
    int N = (int)read_ll(); long long E = read_ll(); int S = (int)read_ll();
    std::vector<int> src(S);
    for (auto& s : src) s = (int)read_ll();
    std::vector<long long> off(N + 1);
    for (auto& o : off) o = read_ll();
    std::vector<int> nb(E), w(E);
    for (auto& x : nb) x = (int)read_ll();
    for (auto& x : w) x = (int)read_ll();
    std::vector<char>().swap(data);

    fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 1; }

    std::vector<int> d(N), tent(N), order;
    std::vector<char> settled(N);
    for (int s : src) {
        // ---- Dijkstra ----
        std::fill(d.begin(), d.end(), INF);
        d[s] = 0;
        using P = std::pair<int, int>;
        std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
        pq.push({0, s});
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (du != d[u]) continue;
            for (long long i = off[u]; i < off[u + 1]; i++) {
                int nd = du + w[i];
                if (nd < d[nb[i]]) { d[nb[i]] = nd; pq.push({nd, nb[i]}); }
            }
        }

        put_int(s);

        // ---- adaptive delta log ----
        if (delta == -1) {
            order.clear();
            for (int v = 0; v < N; v++) if (d[v] != INF) order.push_back(v);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return d[a] < d[b]; });
            std::fill(tent.begin(), tent.end(), INF);
            std::fill(settled.begin(), settled.end(), 0);
            std::set<P> far;  // (tent, vertex) of reached, unsettled vertices
            size_t p = 0;
            long long cutoff = 0;
            std::vector<int> batch;
            while (true) {
                batch.clear();
                while (p < order.size() && d[order[p]] <= cutoff) {
                    int v = order[p++];
                    settled[v] = 1;
                    if (tent[v] != INF) far.erase({tent[v], v});
                    batch.push_back(v);
                }
                for (int u : batch) {
                    for (long long i = off[u]; i < off[u + 1]; i++) {
                        int v = nb[i];
                        if (settled[v]) continue;
                        int nd = d[u] + w[i];
                        if (nd < tent[v]) {
                            if (tent[v] != INF) far.erase({tent[v], v});
                            tent[v] = nd;
                            far.insert({nd, v});
                        }
                    }
                }
                if (far.empty()) break;
                int Dmin = far.begin()->first;
                auto it = far.begin();
                for (int k = 0; k < K - 1 && std::next(it) != far.end(); k++) ++it;
                cutoff = it->first;
                put_int(cutoff - Dmin);
            }
        }

        for (int v = 0; v < N; v++) put_int(d[v]);
    }
    fwrite(obuf, 1, opos, fout);
    fclose(fout);
    return 0;
}
