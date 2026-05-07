a = input("请输入一个整数：")
a = int(a)
print(f"你输入的整数是：{a}")

nums = list(map(int, input("输入多个整数（空格分隔): ").split()))
print(nums)

"""
nums = map(int, input("输入多个整数（空格分隔): ").split())
print(nums)
会返回一个map对象，返回的是迭代器，需要使用list()函数将其转换为列表

nums = input("输入多个整数（空格分隔）: ").split()

imput函数的作用是读取用户在控制台输入的一整行内容，
无论你输入的是数字还是符号，都会被当成字符串

print(nums)
会返回一个列表，列表中的元素是字符串类型

"""
n = int(input("输入行数："))
data = []
# 这里面的data是一个空列表，用来存储输入的所有数

for i in range(n):
    num = int(input(f"请输入第{i+1}个数"))
    data.append(num)
print("输入的所有数：",data)                 