import numpy as np 

class Line:
    def __init__(self, v, *, origin=0, size=1) -> None:
        self.vector = v
        self.size = size
        self.x_values = [i for i in range(0, size, 8)]
        self.y_values = [self.getY(i) for i in self.x_values]
    
    def getXValues(self, size):
        return np.asarray([i for i in range(0, size, 10)])
    
    def getY(self, x):
        x_vector = np.array([x**2, x, 1])
        return np.dot(self.vector, x_vector)
    
    def derivAtX(self, x):
        deriv = np.array([2 * x, 1, 0])
        return np.multiply(self.vector, deriv)
    
    def normVector(self, x):
        dirv = self.derivAtX(x)
        transformMatrix = np.array([[0, -1], [1, 0], [0, 0]])
        return np.matmul(dirv, transformMatrix)