# The Computer Language Benchmarks Game
# http://benchmarksgame.alioth.debian.org/

# transliterated from Mike Pall's Lua program
# contributed by Mario Pernici
# tiny mods by Zach Wegner

N = 10000

i = 0
k = 0
ns = 0
k1 = 1
[n,a,d,t,u] = [1,0,1,0,0]
while(1):
  k = k + 1
  t = n<<1
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
      ns = ns*10 + t
      i = i + 1
      if i % 10 == 0:
        print(str(ns) + '    ' + str(i))
        ns = 0
      if i >= N:
        break
      a = a - d*t
      a = a * 10
      n = n * 10
