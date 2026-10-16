#include <bits/stdc++.h>
using namespace std;

using ll = long long;
using ull = unsigned long long;
using i128 = __int128_t;
using u128 = __uint128_t;
using ld = long double;
using pii = pair<int, int>;
using pll = pair<ll, ll>;
using poly = vector<ll>;
using u64 = uint64_t;
using u128 = __uint128_t;

constexpr ll MOD = 998244353; 
constexpr ll INF = 1LL << 62;
constexpr ll G=3;
const ld PI = acosl(-1.0L);
const ld EPS = 1e-12L;
//const ld INF = numeric_limits<ld>::infinity(); //计算几何

#define Lt Line<T>
#define Pt Point<T>
#define Pd Point<ld>
#define Ld Line<ld>
#define cc(x) cout << fixed << setprecision(x);


template<class T> bool chmin(T &a, const T &b) { 
    return b < a ? a = b, true : false;
}

template<class T> bool chmax(T &a, const T &b) { 
    return a < b ? a = b, true : false;
}

// gcd(|a|,|b|)，允许 a、b 为负数
ll gcdll(ll a, ll b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { ll t = a % b; a = b; b = t; }
    return a;
}

ll norm(ll x, ll mod) { x %= mod; return x < 0 ? x + mod : x; }

// lcm(a,b)，结果可能溢出时自行用 i128 检查
ll lcmll(ll a, ll b) { return a / gcdll(a, b) * b; }

// 返回 g=gcd(a,b)，并求 ax+by=g
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (!b) { x = (a >= 0 ? 1 : -1); y = 0; return a >= 0 ? a : -a; }
    ll x1, y1, g = exgcd(b, a % b, x1, y1);
    x = y1; y = x1 - (a / b) * y1;
    return g;
}

// 解 ax ≡ b (mod m)，返回最小非负解；无解返回 -1
ll linear_mod(ll a, ll b, ll m) {
    a = norm(a, m); b = norm(b, m);
    ll x, y, g = exgcd(a, m, x, y);
    if (b % g) return -1;
    ll mm = m / g;
    return (ll)((i128)x * (b / g) % mm + mm) % mm;
}

ll qpow(ll a, ll b) { 
    ll r = 1;
    for (a = (a % MOD + MOD) % MOD; b; b >>= 1, a = a * a % MOD)
        if (b & 1) r = r * a % MOD;
    return r;
}

ll qpow(ll a, ll b,const ll &mod) { 
    ll r = 1 % mod;
    a %= mod;
    if (a < 0) a += mod;
    for (; b; b >>= 1, a = (ll)((i128)a * a % mod))
        if (b & 1) r = (ll)((i128)r * a % mod);
    return r;
}

// 64 位无溢出乘法，0<=a,b<mod，mod<2^63
ll mul_mod(ll a, ll b, ll mod) { return (ll)((i128)a * b % mod); }

// 质数模逆元：mod 为质数且 a%mod != 0
ll inv_prime(ll a, ll mod) { return qpow(a, mod - 2, mod); }

// 任意模逆元：gcd(a,mod)=1，否则返回 -1
ll inv_any(ll a, ll mod) {
    ll x, y, g = exgcd(a, mod, x, y);
    if (g != 1) return -1;
    return norm(x, mod);
}

// 线性求 1..n 在质数 mod 下的逆元，O(n)
vector<ll> inv_all(ll n, ll mod) {
    vector<ll> iv(n + 1); if (n >= 1) iv[1] = 1;
    for (ll i = 2; i <= n; ++i) iv[i] = (mod - mod / i) * iv[mod % i] % mod;
    return iv;
}

int mex(vector<int> a) {
    int n = (int)a.size();
    vector<int> v(n + 1);
    for (int x : a) if (0 <= x && x <= n) v[x] = 1;
    for (int i = 0; ; ++i) if (!v[i]) return i;
}



struct FastIO {
    static const int S = 1 << 20;
    int ip = 0, in = 0, op = 0;
    char ib[S], ob[S];

    ~FastIO() { flush(); }
    char gc() { // 返回下一字节；EOF 返回 0
        if (ip == in) in = fread(ib, 1, S, stdin), ip = 0;
        return ip == in ? 0 : ib[ip++];
    }
    void pc(char c) { // c：待输出字符
        if (op == S) flush();
        ob[op++] = c;
    }
    void flush() { // 写出当前输出缓冲
        if (op) fwrite(ob, 1, op, stdout), op = 0;
    }

    template<class T> bool read(T &x) { // x：接收整数
        static_assert(is_integral_v<T>);
        char c = gc();
        while (c && c <= ' ') c = gc();
        if (!c) return false;
        bool neg = c == '-';
        if (neg || c == '+') c = gc();
        using U = make_unsigned_t<T>;
        U v = 0;
        while (c >= '0' && c <= '9') v = v * 10 + (c - '0'), c = gc();
        x = neg ? (T)(U(0) - v) : (T)v;
        return true;
    }

    bool read(string &s) { // s：接收一个非空白字符串
        char c = gc();
        while (c && c <= ' ') c = gc();
        if (!c) return false;
        s.clear();
        while (c > ' ') s.push_back(c), c = gc();
        return true;
    }

    template<class T> void write(T x, char end = '\n') {
        // x：待输出整数；end：结尾字符，传 0 表示不输出结尾
        static_assert(is_integral_v<T>);
        using U = make_unsigned_t<T>;
        U v;
        if constexpr (is_signed_v<T>) {
            if (x < 0) pc('-'), v = U(0) - (U)x;
            else v = (U)x;
        } else v = x;
        char s[32]; int n = 0;
        do s[n++] = char('0' + v % 10), v /= 10; while (v);
        while (n) pc(s[--n]);
        if (end) pc(end);
    }
};

u64 mr_mul(u64 a, u64 b, u64 m) { return (u128)a * b % m; }
u64 mr_pw(u64 a, u64 b, u64 m) {
        u64 r = 1 % m;
        for (; b; b >>= 1, a = mr_mul(a, a, m)) if (b & 1) r = mr_mul(r, a, m);
        return r;
    }
bool is_prime(u64 n) { // n：待判断的 unsigned 64 位整数
        if (n < 2) return false;
        for (u64 p : {2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL,29ULL,31ULL,37ULL}) {
            if (n % p == 0) return n == p;
        }
        u64 d = n - 1, s = 0; while (!(d & 1)) d >>= 1, ++s;
        // 这些底数对所有 unsigned 64 位整数确定正确
        for (u64 a : {2ULL,325ULL,9375ULL,28178ULL,450775ULL,9780504ULL,1795265022ULL}) {
            if (a % n == 0) continue;
            u64 x = mr_pw(a % n, d, n); if (x == 1 || x == n - 1) continue;
            bool ok = false;
            for (u64 r = 1; r < s; ++r) { x = mr_mul(x, x, n); if (x == n - 1) { ok = true; break; } }
            if (!ok) return false;
        }
        return true;
    }

// 求 x^2 ≡ n (mod p) 的最小非负根；p 为奇质数，无解返回 -1
ll mod_sqrt(ll n, ll p) {
    n = norm(n, p); if (!n) return 0; if (p == 2) return n;
    if (qpow(n, (p - 1) / 2, p) != 1) return -1; // Euler 判别
    if (p % 4 == 3) { ll x = qpow(n, (p + 1) / 4, p); return min(x, p - x); }
    ll q = p - 1; int s = 0; while (!(q & 1)) q >>= 1, ++s;
    ll z = 2; while (qpow(z, (p - 1) / 2, p) != p - 1) ++z;
    ll c = qpow(z, q, p), x = qpow(n, (q + 1) / 2, p), t = qpow(n, q, p); int m = s;
    while (t != 1) {
        int i = 1; ll tt = mul_mod(t, t, p);
        while (tt != 1) tt = mul_mod(tt, tt, p), ++i;
        ll b = qpow(c, 1LL << (m - i - 1), p);
        x = mul_mod(x, b, p); c = mul_mod(b, b, p); t = mul_mod(t, c, p); m = i;
    }
    return min(x, p - x);
}

#ifdef LOCAL
#define debug(x) cerr << #x << " = " << (x) << '\n'
#else
#define debug(x) ((void)0)
#endif

void solve() 
{




}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T = 1;
    cin >> T; 
    while (T--) solve();
    return 0;
}
