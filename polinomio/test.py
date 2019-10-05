results = []
total = 0
max_value = 256
res = 50
x = 2
target = 13

for i in range(max_value):
    itmp = (i/res)*x*x
    for j in range(max_value):
        jtmp = (j/res)*x
        for k in range(max_value):
            ktmp = k/res
            if itmp + jtmp + ktmp == target:
                results.append([itmp, jtmp, ktmp])
                total += 1

print(results)
print(total)
