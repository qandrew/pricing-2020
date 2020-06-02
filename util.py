import numpy as np
import math
import matplotlib.pyplot as plt

# basic functions

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def catTriangle(w,l):
    return int(math.factorial(w+l)*(w+-l+1)/(math.factorial(l)*math.factorial(w+1)))

def catTrapezoid(w,l,m):
    if 0 <= l < m:
        return nCr(w + l, l)
    elif m <= l <= w + m -1:
        return nCr(w+l,l) - nCr(w+l,l-m)
    else:
        return 0

dp = {}
def catTrapezoidWeighted(w,l,m,a,b):
    f = (w,l,m,a,b)
    if f in dp:
        return dp[f]
    if l == 0:
        return 1
    elif b*l >= a*w + m:
        return 0
    elif w == 0:
        return 1
    else:
        val = catTrapezoidWeighted(w-1,l,m,a,b) + catTrapezoidWeighted(w,l-1,m,a,b)
        dp[f] = val
        return dp[f]
    
dp2 = {}
def catTrapezoidWeightedInverse(x,w,l,p,q,c):
    f = (x,w,l,p,q,c)
    if f in dp2:
        return dp2[f]
    if l == 0:
        return 1
    else:
        x_new = x_update(x,w,l,p,q)
        if local_reward(x_new,p,q,c) < 0:
            dp2[f] = 0
            return dp2[f]
        elif w == 0:
            return 1
        else:
            val = catTrapezoidWeightedInverse(x,w-1,l,p,q,c) + catTrapezoidWeightedInverse(x,w,l-1,p,q,c)
            dp2[f] = val
            return dp2[f]

def getAB(p,q,x):
    a = math.log(p/q)
    b = math.log((1-q)/(1-p))

    return (a/b).as_integer_ratio()
        
def update_prior(x,p,q,result):
    if result == True:
        return p*x/(win(x,p,q))
    else:
        return (1 - p)*x/(lose(x,p,q))

def prior_inverse(x,p,q,result):
    if result == True:
        return x*q/(p - x*p + x*q)
    else:
        return x*(1-q)/( (1-p) - x*(1-p) + x*(1-q))

def local_reward(x,p,q,c):
    out = win(x,p,q) - c
    return out

def win(x,p,q):
    return x*p + (1-x)*q

def lose(x,p,q):
    return x*(1-p) + (1-x)*(1-q)

def x_update(x,w,l,p,q):
    return  (x*p**w*(1-p)**l)/(x*p**w*(1-p)**l + (1-x)*q**w*(1-q)**l)

def expected_profit(x,p,q,c,c2,delta,rounds=100):
    
    # begin: assume c > c2. and that local_reward(x,p,q) > c
    if c < c2:
        return 0
    
    tot = 0
    
    # first, get possible losses
    possibleLosses = 0
    x_new = x
#     print(local_reward(x_new,p,q,c))
    while local_reward(x_new,p,q,c) > 0 and possibleLosses < 40:
        possibleLosses += 1
        x_new = update_prior(x_new,p,q,False)
#     print('I can lose', possibleLosses)
    
    for t in range(rounds):
        lowest = max(0,int((t-possibleLosses+1)/2))
            
        for w in range(lowest,t + 1,1):
            l = t - w
#             print('\n---\n',t,w,l)
#             print('cat',catTrapezoid(w,l,possibleLosses),prob_reach(x,w,l,p,q))
            add = delta**t*(c - c2)*catTrapezoid(w,l,possibleLosses+1)*prob_reach(x,w,l,p,q)
#             print('add', catTrapezoid(w,l,possibleLosses+1),prob_reach(x,w,l,p,q),add)
            tot += add
    return tot

def expected_profit_weighted(x,p,q,c,c2,delta,rounds=100):
    print(p,q,c)
    
    # begin: assume c > c2. and that local_reward(x,p,q) > c
    if c < c2:
        return 0
    
    tot = 0

    for t in range(rounds):
        lowest = 0
            
        for w in range(lowest,t + 1,1):
            l = t - w
#             print(f"t: {t}, w: {w}, l: {l}, paths: {catTrapezoidWeightedInverse(x,w,l,p,q,c)}")
#             print(x,w,l,p,q,c)
            add = delta**t*(c - c2)*catTrapezoidWeightedInverse(x,w,l,p,q,c)*prob_reach(x,w,l,p,q)
            tot += add
    return tot

def prob_reach(x,w,l,p,q):
    prob = 1
    for i in range(w):
        prob *= win(x,p,q)
        x = update_prior(x,p,q,True)
    for i in range(l):
        prob *= lose(x,p,q)
        x = update_prior(x,p,q,False)
    return prob

def getPossibleCX(x,p,q,c):
    # todo: add c2
    # returns the possible static prices such that local reward > 0
    if local_reward(x,p,q,c) < 0:
        return []
    elif x == 1:
        return []
    else:
        kList = []
        possible = True
        k = 0
        while possible: # todo if c < q then forever
            if local_reward(x,p,q,c) <= 0 or k > 30:
                possible = False
            k += 1
            c2 = win(x,p,q)
            kList.append(c2)
#             print(c2, k)
            x = update_prior(x,p,q,False)
        return kList
