n = int(input())

arr = list(map(int, input().split()))

for j in range(n):
    for i in range(n - 1):
        if arr[i] > arr[i + 1]:
            arr[i + 1] += arr[i] - arr[i + 1]
            arr[i] -= arr[i] - arr[i + 1]
        else:
            continue

print(*arr)