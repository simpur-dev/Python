import sys

def main():
    n = int(sys.stdin.readline())
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2

    if n <= 2:
        print(n)

    return

    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    print(dp[n])

if __name__ == "__main__":
    main()