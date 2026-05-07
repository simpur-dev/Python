from collections import Counter

s = "banana"

#相当于自动帮我做了
# cnt = {}
# for ch in s:
#   cnt[ch] += 1

cnt = Counter(s)

print(cnt)
print(cnt["a"])
print(cnt["b"])
print(cnt["n"])
print(cnt["x"])

print(dict(cnt))

nums = [1, 2, 2, 3, 3, 3, 4]

cnt2 = Counter(nums)

print(cnt2)
print(cnt2[1])
print(cnt2[2])
print(cnt2[3])
print(cnt2[3])
print(cnt2[5])

print(cnt2.most_common(2))