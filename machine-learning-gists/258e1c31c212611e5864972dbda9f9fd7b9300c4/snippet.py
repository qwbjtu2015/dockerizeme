"""This is machine learning cource
    I first of all start with numpy"""
import numpy as np
print(np.__version__)

L = list(range(10))
[str(c) for c in L]
for c in L:
    print(str(c))

print(np.ones((3,5), dtype=float))
print(np.full((3,5),1.5))
print(np.arange(0,20,2))
print(np.linspace((0,10,5))
print(np.random.normal(1,2,(3,3)))
print(np.eye(4))


print(np.random.seed(0))
x1= np.random.randint(10,size=3)
x2= np.random.randint(20,size=(3,4))
x3= np.random.randint(30,size=(3,4,5))
print("x3 ndim:", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size:", x3.size)

x4=np.array([4,2,5,3,4,8])
print(x1[0])   #get in first
print(x1[-1])     #get from last
x5=np.array([[1,4,2,5],
            [5,2,3,4]])
print(x5[1,3])


""" now look for array slicing"""
x = np.arange(10)
print(x)
print(x[:5])
print(x[1::3])
print(x[2::])
print(x[2:])
print(x[::5])
print(x[-1:])
print(x[:-2])
print(x[::-1])

""" Array concatenation """

x = np.array([1,2,3])
y = np.array([3,2,1])
z = [21,21,21]
print(np.concatenate([x,y,z]))
grid = np.array([[10,11,12],[13,14,15]])
print(np.concatenate([grid,grid]))             #use as make 2D array
print(np.concatenate([grid,grid],axis=1))        #use for make axis as x '1' and y='0'

x = np.arange(10)
x1,x2,x3 = np.split(x,[3,6])
print(x1,x2,x3)

grid = np.arange(16).reshape((4,4))
upper,lower = np.split(grid,[2])
print(upper,lower)
print(grid)