# P1219 [USACO1.5] 八皇后 Checker Challenge
import sys

n = int(sys.stdin.readline())
a = [0] * n
col = [False] * n
diag1 = [False] * (2 * n)
diag2 = [False] * (2 * n)
cnt = 0

def dfs(row):
    global cnt
    if row == n:
        cnt += 1
        if cnt <= 3:
            print(' '.join(str(c + 1) for c in a))
        return

    for c in range(n):
        if not col[c] and not diag1[row - c + n] and not diag2[row + c]:
            a[row] = c
            col[c] = diag1[row - c + n] = diag2[row + c] = True
            dfs(row + 1)
            col[c] = diag1[row - c + n] = diag2[row + c] = False
            print(f"回溯: row={row}, 列{c+1} 试完, 返回上一层")

dfs(0)
print(cnt)