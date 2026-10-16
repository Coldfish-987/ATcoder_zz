## 6. Gomory-Hu Tree：无向图任意两点最小割

对带非负容量的无向图构造一棵割树。树上任意两点路径的最小边权，等于原图中这两点之间的最小割。构树需要 `n-1` 次最大流；完成倍增后单次查询 `O(log n)`。

- `GH g(n,m)`：建立无向图，`m` 为无向边数预估。
- `add(u,v,c)`：加入容量为 `c` 的无向边。
- `build()`：构造割树和倍增表，只调用一次。
- `ask(u,v)`：查询不同点 `u,v` 之间的最小割。
- `tr`：最终割树邻接表，可用于离线处理全部询问。

```cpp
struct GH {
    int n, lg;
    Dinic f;
    vector<int> p, dep;
    vector<ll> w;
    vector<vector<pair<int, ll>>> tr;
    vector<vector<int>> up;
    vector<vector<ll>> mn;

    GH(int n, int m = 0) : n(n), f(n, m), tr(n + 1) {}
    void add(int u, int v, ll c) { // u-v: 无向边; c: 容量
        f.add(u, v, c, c);
    }
    void build() {
        p.assign(n + 1, 1), w.assign(n + 1, 0);
        p[1] = 0;
        for (int s = 2; s <= n; s++) {
            int t = p[s];
            f.reset();
            ll z = f.maxflow(s, t);
            auto vis = f.cut(s);
            for (int i = s + 1; i <= n; i++)
                if (p[i] == t && vis[i]) p[i] = s;
            if (vis[p[t]]) {
                p[s] = p[t], p[t] = s;
                w[s] = w[t], w[t] = z;
            } else {
                w[s] = z;
            }
        }
        tr.assign(n + 1, {});
        for (int i = 2; i <= n; i++) {
            tr[i].push_back({p[i], w[i]});
            tr[p[i]].push_back({i, w[i]});
        }

        lg = 1;
        while ((1 << lg) <= n) ++lg;
        up.assign(lg, vector<int>(n + 1));
        mn.assign(lg, vector<ll>(n + 1, INF));
        dep.assign(n + 1, -1);
        queue<int> q;
        dep[1] = 0, q.push(1);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto [v, c] : tr[u]) if (v != up[0][u]) {
                dep[v] = dep[u] + 1;
                up[0][v] = u, mn[0][v] = c;
                q.push(v);
            }
        }
        for (int k = 1; k < lg; k++) {
            for (int v = 1; v <= n; v++) {
                up[k][v] = up[k - 1][up[k - 1][v]];
                mn[k][v] = min(mn[k - 1][v], mn[k - 1][up[k - 1][v]]);
            }
        }
    }
    ll ask(int u, int v) const { // u,v: 两个不同点
        ll ans = INF;
        if (dep[u] < dep[v]) swap(u, v);
        int d = dep[u] - dep[v];
        for (int k = 0; k < lg; k++) if (d >> k & 1)
            ans = min(ans, mn[k][u]), u = up[k][u];
        if (u == v) return ans;
        for (int k = lg - 1; k >= 0; k--) if (up[k][u] != up[k][v]) {
            ans = min({ans, mn[k][u], mn[k][v]});
            u = up[k][u], v = up[k][v];
        }
        return min({ans, mn[0][u], mn[0][v]});
    }
};
```

平行边可以直接加入，会自然累加割容量；自环对割无贡献，可忽略。图不连通时也可构造，跨连通块的查询答案为 0。`ask(u,u)` 没有实际意义，本模板要求两点不同。

如果只需无向图的全局最小割，而不是任意点对询问，直接使用 Stoer-Wagner 通常更省代码和时间。