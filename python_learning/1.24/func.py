def print_info(**kwargs):
    print(f"传入的信息：{kwargs}")
    for key, value in kwargs.items():  # kwargs是字典，可遍历键值对
        print(f"{key}：{value}")

# 调用：传入不同个数的关键字参数
print_info(name="张三", age=20)
print_info(name="李四", age=18, height=165, weight=50)
print_info()