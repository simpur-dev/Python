from collections import deque

grid =[
    ".....",
    ".###.",
    ".#...",
    ".###.",
    "....."
]

n = 5
m = 5

dist = [[-1] * m for _ in range(n)]

q = deque()
q.append((0,0))
dist[0][0] = 0

dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

while q:
    x, y = q.popleft()

    for dx, dy in dirs:
        nx = x + dx
        ny = y + dy

        if 0 <= nx < n and 0 <= ny < m:
            if grid[nx][ny] != "#" and dist[nx][ny] == -1:
                # 为什么一个点没有必要走第二次：因为再次走到同一个点，只可能有两种情况：
                # 步数一样，第二条路线不会让答案更优
                # 步数更长
                dist[nx][ny] = dist[x][y] + 1
                q.append((nx, ny))

print(dist[n-1][m-1])