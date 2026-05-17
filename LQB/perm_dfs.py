n = 3
a = [0] * n
vis = [False] * (n + 1)

def dfs(depth):
    if depth == n:
        print(a)
        return
    
    for i in range(1, n + 1):
        if not vis[i]:
            vis[i] = True
            a[depth] = i
            dfs(depth + 1)
            vis[i] = False

def main():
    dfs(0)


if __name__ == "__main__":
    main()