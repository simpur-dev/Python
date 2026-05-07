def gvd(a, b):
    while b:
        a, b = b, a % b
    return a

a = int(input("请输入第一个数："))
b = int(input("请输入第二个数："))
result = gvd(a, b)
print(f"{a} 和 {b} 的最大公约数是：{result}")

