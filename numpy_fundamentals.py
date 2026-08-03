import numpy as np

python_list = [1,2,3,4,5]
numpy_array = np.array([1,2,3,4,5])
python_list = [x+10 for x in python_list]
numpy_array = (numpy_array + 10)


arr1 = np.arange(0,10,2)
arr2 = np.linspace(0,12,4)
arr3 = np.eye(3)
arr4 = np.full(5,7)
arr5 = np.random.rand(5)
arr6 = np.random.randint(0,10,5)

ar1 = np.array([1,2,3])
ar2 = np.array([1.2, 3.4, 5.6])
ar3 = np.array([1, 2.5, 3])

m1 = np.array([[1,2,3], [4,5,6], [7,8,9]])
m2 = np.array([[[1,2,3], [3,4,5]], [[4,5,6], [4,2,1]]])
'''
print(m2)
print(m2.shape)
print(m2.ndim)
print(m2.size)
'''
m1 = np.array([[1,2,3],[4,5,6]])
'''
print(m1[0])
print(m1[1,2])
print(m1[:,1])
print(m1[0:1, 0:2])
'''

arr = np.array([10, 20, 30, 40, 50])
mask = arr > 25
'''
print(mask)
print(arr[mask])
'''

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
'''
print(a + b)
print(a * b)
print(a + 5)
print(a * 2)
'''

arr = np.array([1,2,3,4,5,6])
'''
print(arr.reshape(2,3))
print(arr.reshape(3,2))
print(arr.reshape(2,4))  
'''

m = np.array([[1,2,3],[4,5,6]])
'''
print(m.sum())
print(m.sum(axis=0))
print(m.sum(axis=1))
'''

arr = np.array([2, 4, 4, 4, 5, 5, 7, 9])
'''
print(arr.mean())
print(arr.std())
print(arr.var())
print(np.median(arr))
print(arr.min(), arr.max())
'''

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

#print(np.dot(a, b))

a = [[1,2], [3,4]]
b = [[5,6], [7,8]]

result = []
for i in range(len(a)):
    row = []
    for j in range(len(b[0])):
        total = 0
        for k in range(len(a[0])):
            total += a[i][k] * b[k][j]
        row.append(total)
    result.append(row)

c = np.array([[1,2,3], [4,5,6], [7,0,9]])
#print(c)
#print(np.linalg.det(c))
#print(c.T)
#print(np.linalg.inv(c))
#print(c @ np.linalg.inv(c))


arr1 = np.array([3,1,4,1,5])
result = np.sort(arr1)
#print(arr1)      
#print(result)     

arr2 = np.array([3,1,4,1,5])
x = arr2.sort()
#print(arr2)      
#print(x)          


a = np.array([1,2,3])
b = np.array([4,5,6])

print(np.vstack((a,b)))
print(np.hstack((a,b)))
print(np.column_stack((a,b)))