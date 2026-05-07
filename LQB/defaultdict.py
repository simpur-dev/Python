from collections import defaultdict

#defaultdict(int)表示不存在的键默认是0
cnt = defaultdict(int)

a = "abca"

for ch in a:
    cnt[ch] += 1

print(cnt)

print(cnt["a"])
print(cnt["b"])

#普通字典如果访问不存在的键，会报错
# d = {}
# print(d["a"])


groups = defaultdict(list)


students = [
    ("Alice", "A"),
    ("Bob", "B"),
    ("Cindy", "A"),
    ("David", "B")
]

for name, cls in students:
    groups[cls].append(name)

print(groups)
print(dict(groups))
print(groups["A"])
print(groups["B"])
