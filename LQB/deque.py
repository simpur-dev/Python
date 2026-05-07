from collections import deque

q = deque()

q.append(10)
q.append(20)
q.append(30)

print(q)

x = q.popleft()
print(x)
print(q)

q.append(40)

print(q)

while q:
    y = q.popleft()
    print(y)


