from collections import deque

start = 1
target = 10

q = deque()
q.append(start)

dist = {}
# dist[start] = 0
#不会报错，因为这是在给字典新增内容

# dist = {}
# print(dist[2])
# 普通字典访问不存在的键会报错


while q:
    x = q.popleft()

    if x == target:
        print(dist[x])

        break

    for nx in [x + 1, x * 2]:
        if nx <= 100 and nx not in dist:
            dist[nx] = dist[x] + 1
            q.append(nx)