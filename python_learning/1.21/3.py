for i in range(5):
    print(i , end=" ")
print("\n")

nums = [1,2,3,4,5]
for num in nums:
    print(num * 2, end=" ")
print("\n")

i = 0
while i < 5:
    if i % 2 == 0:
        print(f"{i}是偶数")
    else:
        print(f"{i}是奇数")
    i += 1

a, b = 3, 5
res = a if a > b else b
print("较大值：", res)

