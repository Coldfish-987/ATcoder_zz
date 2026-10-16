# 凸多边形每个顶点的最远点：SMAWK

默认 C++17。给定严格凸多边形 `p[0..n-1]`，按逆时针排列且不重复首点，求每个顶点 `i` 的一个最远顶点 `far[i]`。距离比较使用平方距离，不开根号。代码复用《计算几何_二维.md》的 `Point<T>`；复制该文档的 0、1 节后再复制这里的代码。

## 1. 板子

循环距离矩阵展开为 `n` 行、`2n-1` 列，第 `i` 行只允许列 `j in [i,i+n-1]`，实际顶点为 `p[j%n]`。无效位置填充有序的负无穷，保持阶梯边界的全单调性。SMAWK 不建立 `O(n^2)` 矩阵，整体 `O(n)` 时间、`O(n)` 额外空间。

并列最远时返回展开下标较大的一个，即沿逆时针方向更远的那个。

```cpp
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using i128 = __int128_t;

i128 dist2(Point<ll> a, Point<ll> b) { // 复用二维板子的 Point<T>
    i128 x = (i128)a.x - b.x, y = (i128)a.y - b.y;
    return x*x + y*y;
}

template<class Better>
vector<int> smawk(const vector<int>& rows, const vector<int>& cols,
                  Better better) {
    if (rows.empty()) return {};
    vector<int> reduced;
    for (int c : cols) {
        while (!reduced.empty()
            && better(rows[(int)reduced.size()-1], reduced.back(), c))
            reduced.pop_back();
        if ((int)reduced.size() < (int)rows.size()) reduced.push_back(c);
    }
    vector<int> oddRows;
    for (int i=1; i<(int)rows.size(); i+=2) oddRows.push_back(rows[i]);
    vector<int> oddAns = smawk(oddRows, reduced, better);

    vector<int> ans(rows.size());
    int k = 0;
    for (int i=0; i<(int)rows.size(); i+=2) {
        int L = i == 0 ? 0 : (int)(lower_bound(reduced.begin(), reduced.end(),
                                               oddAns[k-1]) - reduced.begin());
        int R = i+1 == (int)rows.size() ? (int)reduced.size()-1
                                        : (int)(lower_bound(reduced.begin(), reduced.end(),
                                                            oddAns[k]) - reduced.begin());
        int best = reduced[L];
        for (int z=L+1; z<=R; ++z)
            if (better(rows[i], best, reduced[z])) best = reduced[z];
        ans[i] = best;
        if (i+1 < (int)rows.size()) ans[i+1] = oddAns[k++];
    }
    return ans;
}

vector<int> farthestVertices(const vector<Point<ll>>& p) {
    int n = (int)p.size();
    if (n == 0) return {};
    if (n == 1) return {0};
    if (n == 2) return {1,0};

    const i128 NEG = -((i128)1 << 120);
    int m = 2*n-1;
    vector<int> rows(n), cols(m);
    iota(rows.begin(), rows.end(), 0);
    iota(cols.begin(), cols.end(), 0);

    auto value = [&](int i, int j) -> i128 {
        if (j < i) return NEG - 4*n + j; // 左无效区：从左至右递增
        if (j >= i+n) return NEG - j;     // 右无效区：从左至右递减
        return dist2(p[i], p[j%n]);
    };
    auto better = [&](int i, int a, int b) {
        return value(i,a) <= value(i,b);
    };

    vector<int> pos = smawk(rows, cols, better);
    vector<int> ans(n);
    for (int i=0; i<n; ++i) ans[i] = pos[i] % n;
    return ans;
}
```

用法：

```cpp
vector<Point<ll>> p = {{0,0},{4,0},{5,2},{2,5},{-1,3}};
auto far = farthestVertices(p);
for (int i=0; i<(int)p.size(); ++i)
    cout << i << " -> " << far[i] << '\n';
```

## 2. 正确性与复杂度

对有效位置 `i <= j < i+n`，若 `i1<i2`、`j1<j2` 且四点按凸多边形边界顺序排列，则有交叉不等式：

```text
dist(i1,j1) <= dist(i1,j2)  =>  dist(i2,j1) <= dist(i2,j2).
```

比较两个距离平方时，公共项消去，剩下的是点到弦 `p[j1]p[j2]` 的有向投影；沿另一条凸边界链移动，投影单调。这说明每行最优列不会向左跳，即有效部分全单调。左、右无效阶梯分别按“向有效区方向更优”的顺序放置负无穷，跨越边界后仍保持全单调。

SMAWK 每层列规约扫描一次当前列集，递归处理奇数行，偶数行只扫描两个相邻奇数行答案之间的列；总比较次数 `O(n)`，每个单元的距离计算为 `O(1)`。

## 3. 注意事项

- 输入必须是严格凸、同向排列的多边形。无序点集先使用二维板子的 `convexHull(p,true)`；顺时针输入先 `reverse`。
- `n=1,2` 已单独处理；通常题目保证 `n>=3`。
- 坐标差和平方在 `i128` 中计算。若坐标绝对值接近 `1e18`，平方距离可能超过 `i128`，需按题目范围改用更宽整数或高精度数。
- 只求距离时可计算 `dist2(p[i],p[far[i]])`；不需要开根号。
- 并列最远点返回其中一个。若必须取最小下标，需要重新设计并列比较，不能只把 `<=` 改成 `<` 而忽略循环展开后的列顺序。
