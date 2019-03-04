import numpy as np

x = np.array([[  [1,2,3],
                [4,5,6],
                [7,8,9]],
             [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
])

y = np.array([  [1,2,3],
                [4,5,6],
                [7,8,9]])

foo = x.flatten('K')
print(foo)

print(np.average(x))
#print(np.roll(x.flatten(),3))
#print(x.T)
#print(x.T.flatten())

#print(np.dot(2,x.flatten()))

print(int(2.99))
x = [1,2,3,4,5]
print(x[0:2])

for i in range(0,100,3):
    print(i)