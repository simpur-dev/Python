n = 7
k = 4
a = [0] * (k)

def dfs(depth, start):
    if depth == k:
        print(a)
        return
    
    for i in range(start, n + 1):
        a[depth] = i
        dfs(depth + 1, i + 1)

def main():
    dfs(0, 2)

if __name__ == "__main__":
    main()