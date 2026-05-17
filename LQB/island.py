# ## ：岛屿计数（连通块）
# ### 题目描述
# 给定一个 n × m 的字符矩阵：

# - 1 — 陆地
# - 0 — 水域
# 一块「岛屿」由 上下左右 相邻的陆地连通组成。请你数出矩阵中一共有多少座岛屿。

# ### 输入格式
# 第一行两个整数 n 和 m （1 ≤ n, m ≤ 1000）。

# 接下来 n 行，每行一个长度为 m 的字符串，只包含 0 和 1 。

# ### 输出格式
# 一个整数，岛屿数量。

import sys
from collections import deque
sys.setrecursionlimit(1000000)

n, m = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(n)]

vis = [[False] * m for _ in range(n)]
dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
ans = 0

def dfs(x, y):
    vis[x][y] = True
    for dx, dy in dirs:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < n and 0 <= ny < m:
            if grid[nx][ny] == '1' and not vis[nx][ny]:
                dfs(nx, ny)

def main():
    global ans
    for x in range(n):
        for y in range(m):
            if grid[x][y] == '1' and not vis[x][y]:
                ans += 1
                dfs(x, y)
    print(ans)

if __name__ == "__main__":
    main()