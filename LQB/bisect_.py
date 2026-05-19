import bisect

def main():
    arr = [1, 4, 6, 8, 10]
    print(bisect.bisect_right(arr, 5))

    print(bisect.bisect_left(arr, 11))


    print(bisect.bisect_left(arr, 4))

if __name__ == "__main__":
    main()