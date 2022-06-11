import numpy as np 

class PRNG(object):
    """Base class for any Pseudo-Random Number Generator."""
    def __init__(self, X0=0):
        """Create a new PRNG with seed X0."""
        self.X0 = X0
        self.X = X0
        self.t = 0
        self.max = 0
    
    def __iter__(self):
        """self is already an iterator!"""
        return self
    
    def seed(self, X0=None):
        """Reinitialize the current value with X0, or self.X0.
        
        - Tip: Manually set the seed if you need reproducibility in your results.
        """
        self.t = 0
        self.X = self.X0 if X0 is None else X0
    
    def __next__(self):
        """Produce a next value and return it."""
        # This default PRNG does not produce random numbers!
        self.t += 1
        return self.X
    
    def randint(self, *args, **kwargs):
        """Return an integer number in [| 0, self.max - 1 |] from the PRNG."""
        return np.int64(self.__next__())

    def int_samples(self, shape=(1,)):
        """Get a numpy array, filled with integer samples from the PRNG, of shape = shape."""
        # return [ self.randint() for _ in range(size) ]
        return np.fromfunction(np.vectorize(self.randint), shape=shape, dtype=int)

    def rand(self, *args, **kwargs):
        """Return a float number in [0, 1) from the PRNG."""
        return self.randint() / float(1 + self.max)

    def float_samples(self, shape=(1,)):
        """Get a numpy array, filled with float samples from the PRNG, of shape = shape."""
        # return [ self.rand() for _ in range(size) ]
        return np.fromfunction(np.vectorize(self.rand), shape=shape, dtype=int)

# 
class LinearCongruentialGenerator(PRNG):
    """A simple linear congruential Pseudo-Random Number Generator."""
    def __init__(self, m, a, c, X0=0):
        """Create a new PRNG with seed X0."""
        super(LinearCongruentialGenerator, self).__init__(X0=X0)
        self.m = self.max = m
        self.a = a
        self.c = c
    
    def __next__(self):
        """Produce a next value and return it, following the recurrence equation: X_{t+1} = (a X_t + c) mod m."""
        self.t += 1
        x = self.X
        self.X = (self.a * self.X + self.c) % self.m
        return x


m = 1 << 31 - 1  # 1 << 31 = 2**31
a = 7 ** 4
c = 0

SecondExample = LinearCongruentialGenerator(m=m, a=a, c=c, X0=12011993)


shape = (400, 400)
image = SecondExample.float_samples(shape)
image1 = SecondExample.rand()
print(image1)

# %matplotlib inline
import matplotlib.pyplot as plt

def showimage(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.show()

showimage(image)

# 
class MultipleRecursiveGenerator(PRNG):
    """A Multiple Recursive Pseudo-Random Number Generator (MRG), with one sequence (X_t)."""
    def __init__(self, m, a, X0):
        """Create a new PRNG with seed X0."""
        assert np.shape(a) == np.shape(X0), "Error: the weight vector a must have the same shape as X0."
        super(MultipleRecursiveGenerator, self).__init__(X0=X0)
        self.m = self.max = m
        self.a = a
    
    def __next__(self):
        """Produce a next value and return it, following the recurrence equation: X_t = (a_1 X_{t-1} + ... + a_k X_{t-k}) mod m."""
        self.t += 1
        x = self.X[0]
        nextx = (np.dot(self.a, self.X)) % self.m
        self.X[1:] = self.X[:-1]
        self.X[0] = nextx
        return x

m = (1 << 31) - 1
X0 = np.array([10, 20, 30])
a = np.array([10, 9, 8])

ThirdExample = MultipleRecursiveGenerator(m, a, X0)

shape = (400, 400)
image = ThirdExample.float_samples(shape)

showimage(image)


# 
class CombinedMultipleRecursiveGenerator(PRNG):
    """A Multiple Recursive Pseudo-Random Number Generator (MRG), with two sequences (X_t, Y_t)."""
    def __init__(self, m1, a, X0, m2, b, Y0):
        """Create a new PRNG with seeds X0, Y0."""
        assert np.shape(a) == np.shape(X0), "Error: the weight vector a must have the same shape as X0."
        assert np.shape(b) == np.shape(Y0), "Error: the weight vector b must have the same shape as Y0."
        self.t = 0
        # For X
        self.m1 = m1
        self.a = a
        self.X0 = self.X = X0
        # For Y
        self.m2 = m2
        self.b = b
        self.Y0 = self.Y = Y0
        # Maximum integer number produced is max(m1, m2)
        self.m = self.max = max(m1, m2)
    
    def __next__(self):
        """Produce a next value and return it, following the recurrence equation: X_t = (a_1 X_{t-1} + ... + a_k X_{t-k}) mod m."""
        self.t += 1
        # For X
        x = self.X[0]
        nextx = (np.dot(self.a, self.X)) % self.m1
        self.X[1:] = self.X[:-1]
        self.X[0] = nextx
        # For Y
        y = self.Y[0]
        nexty = (np.dot(self.b, self.Y)) % self.m2
        self.Y[1:] = self.Y[:-1]
        self.Y[0] = nexty
        # Combine them
        u = x - y + (self.m1 if x <= y else 0)
        return u

m1 = (1 << 32) - 209                  # important choice!
a = np.array([0, 1403580, -810728])   # important choice!
X0 = np.array([1000, 10000, 100000])  # arbitrary choice!

m2 = (1 << 32) - 22853                # important choice!
b = np.array([527612, 0, -1370589])   # important choice!
Y0 = np.array([5000, 50000, 500000])  # arbitrary choice!

MRG32k3a = CombinedMultipleRecursiveGenerator(m1, a, X0, m2, b, Y0)

shape = (400, 400)
image = MRG32k3a.float_samples(shape)

showimage(image)

from datetime import datetime

def get_seconds():
    d = datetime.today().timestamp()
    s = 1e6 * (d - int(d))
    return int(s)

def seed_rows(example, n, w):
    return example.int_samples((n,))

def random_Mersenne_seed(n, w):
    linear = LinearCongruentialGenerator(m=(1 << 31) - 1, a=7 ** 4, c=0, X0=get_seconds())
    assert w == 32, "Error: only w = 32 was implemented"
    m1 = (1 << 32) - 209                  # important choice!
    a = np.array([0, 1403580, -810728])   # important choice!
    X0 = np.array(linear.int_samples((3,)))  # random choice!
    m2 = (1 << 32) - 22853                # important choice!
    b = np.array([527612, 0, -1370589])   # important choice!
    Y0 = np.array(linear.int_samples((3,)))  # random choice!
    MRG32k3a = CombinedMultipleRecursiveGenerator(m1, a, X0, m2, b, Y0)
    seed = seed_rows(MRG32k3a, n, w)
    assert np.shape(seed) == (n,)
    return seed

example_seed = random_Mersenne_seed(n, w)

class MersenneTwister(PRNG):
    """The Mersenne twister Pseudo-Random Number Generator (MRG)."""
    def __init__(self, seed=None,
                 w=32, n=624, m=397, r=31,
                 a=0x9908B0DF, b=0x9D2C5680, c=0xEFC60000,
                 u=11, s=7, v=15, l=18):
        """Create a new Mersenne twister PRNG with this seed."""
        self.t = 0
        # Parameters
        self.w = w
        self.n = n
        self.m = m
        self.r = r
        self.a = a
        self.b = b
        self.c = c
        self.u = u
        self.s = s
        self.v = v
        self.l = l
        # For X
        if seed is None:
            seed = random_Mersenne_seed(n, w)
        self.X0 = seed
        self.X = np.copy(seed)
        # Maximum integer number produced is 2**w - 1
        self.max = (1 << w) - 1
        
    def __next__(self):
        """Produce a next value and return it, following the Mersenne twister algorithm."""
        self.t += 1
        # 1. --- Compute x_{t+n}
        # 1.1.a. First r bits of x_t : left = (x_t >> (w - r)) << (w - r)
        # 1.1.b. Last w - r bits of x_{t+1} : right = x & ((1 << (w - r)) - 1)
        # 1.1.c. Concatenate them together in a binary vector x : x = left + right
        left = self.X[0] >> (self.w - self.r)
        right = (self.X[1] & ((1 << (self.w - self.r)) - 1))
        x = (left << (self.w - self.r)) + right
        xw = x % 2             # 1.2. get xw
        if xw == 0:
            xtilde = (x >> 1)            # if xw = 0, xtilde = (x >> 1)
        else:
            xtilde = (x >> 1) ^ self.a   # if xw = 1, xtilde = (x >> 1) ⊕ a
        nextx = self.X[self.m] ^ xtilde  # 1.3. x_{t+n} = x_{t+m} ⊕ \tilde{x}
        # 2. --- Shift the content of the n rows
        oldx0 = self.X[0]          # 2.a. First, forget x0
        self.X[:-1] = self.X[1:]   # 2.b. shift one index on the left, x1..xn-1 to x0..xn-2
        self.X[-1]  = nextx        # 2.c. write new xn-1
        # 3. --- Then use it to compute the answer, y
        y = nextx                      # 3.a. y = x_{t+n}
        y ^= (y >> self.u)             # 3.b. y = y ⊕ (y >> u)
        y ^= ((y << self.s) & self.b)  # 3.c. y = y ⊕ ((y << s) & b)
        y ^= ((y << self.v) & self.c)  # 3.d. y = y ⊕ ((y << v) & c)
        y ^= (y >> self.l)             # 3.e. y = y ⊕ (y >> l)
        return y

ForthExample = MersenneTwister(seed=example_seed)

shape = (400, 400)
image = ForthExample.float_samples(shape)

showimage(image)
