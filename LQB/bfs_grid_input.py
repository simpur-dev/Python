# ## 🧩 题目：迷宫最短路径
# ### 题目描述
# 给定一个 n × m 的字符矩阵迷宫：

# - S — 起点
# - E — 终点
# - . — 空地，可以走
# - # — 墙壁，不能走
# 每次可以向 上、下、左、右 四个方向移动一步。请你求出从起点 S 到终点 E 的 最短步数 。如果无法到达终点，输出 -1 。




import sys
from collections import deque

n, m = map(int, sys.stdin.readline().split())

grid = []

for _ in range(n):
    row = sys.stdin.readline().strip()
    grid.append(row)

dist = [[-1] * m for _ in range(n)]

q = deque()
for i in range(n):
    for j in range(m):
        if grid[i][j] == 'S':
            q.append((i,j))
            dist[i][j] = 0

dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

while q:
    x, y = q.popleft()

    if grid[x][y] == 'E':
        print(dist[x][y])
        sys.exit(0)

    for dx, dy in dirs:
        nx = x + dx
        ny = y + dy

        if 0 <= nx < n and 0 <= ny < m:
            if grid[nx][ny] != '#' and dist[nx][ny] == -1:
                dist[nx][ny] = dist[x][y] + 1
                q.append((nx, ny))

print(-1)