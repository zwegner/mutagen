# The Computer Language Benchmarks Game
# http://benchmarksgame.alioth.debian.org/

# transliterated from Mike Pall's Lua program
# contributed by Mario Pernici
# tiny mods by Zach Wegner

N = 10000

i = k = ns = a = t = u = 0
k1 = n = d = 1
while True:
    k = k + 1
    t = n << 1
    n = n * k
    a = a + t
    k1 = k1 + 2
    a = a * k1
    d = d * k1
    if a >= n:
        x = n * 3 + a
        t = x // d
        u = x % d + n
        if d > u:
            ns = ns * 10 + t
            i = i + 1
            if i % 10 == 0:
                print('{:0>10}    {}'.format(ns, i))
                ns = 0
            if i >= N:
                break
            a = a - d * t
            a = a * 10
            n = n * 10
