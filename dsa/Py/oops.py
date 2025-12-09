a = int(input())
b = int(input())

def sum(a, b):
    return a + b

def difference(a, b):
    return a - b

def product(a, b):
    return a * b

def quotient(a, b):
    return a / b

print(sum(a, b))
print(difference(a, b))
print(product(a, b))
print(quotient(a, b))


length = int(input())
width = int(input())

def area(l, w):
    return (l * w)

print(area(length, width))


m1 = int(input())
m2 = int(input())
m3 = int(input())

def avg(m1, m2, m3):
    return (m1 + m2 + m3) / 3

print(avg(m1, m2, m3))

c = float(input())

def faren(c):
    return (c * 9 / 5) + 32

print(faren(c))

s = input()

def palindrome(s):
    if s == s[::-1]:
        print("Palindrome")

    else:
        print("Not a Palindrome")

s = input()

def vow_cou(s):
    c = 0
    for ch in s:
        if ch in "aeiouAEIOU":
            c += 1

    return c

print(vow_cou(s))

balance = 0

def deposit(money):
    balance += money

def withdraw(money):
    balance -= money

def balance():
    return balance

deposit(5000)
withdraw(3000)
print(balance())

n = int(input())

def shoppin_bill(n):
    total = 0

    for i in range(n):
        total += float(input())

    print(total)

shoppin_bill(n)