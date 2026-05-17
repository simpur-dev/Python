from itertools import permutations

def main():
    for p in permutations([1, 2, 3]):
        print(type(p), p)


    print("---")

    for p in permutations([1, 2, 3, 4], 2):
        print(type(p), p)
        p = list(p)
        print(type(p), p)

if __name__ == "__main__":
    main()
