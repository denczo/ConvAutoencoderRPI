import numpy as np
from scipy.sparse import csr_matrix

m1 = np.zeros((5,10))
m2 = np.zeros(10)

m2[2] = 1
m2[5] = 5
for i in range(len(m2)):
    m2[i] = i
m2.shape = (1,10)

for i in range(len(m1)):
    m1[i][i] = i+2

print(m1,m2)
print(np.dot(m1,m2.T))

test = csr_matrix(m1).dot(csr_matrix(m2.T))
print(test.toarray())

#print(csr_matrix(m1).dot(csr_matrix(m2.T)))