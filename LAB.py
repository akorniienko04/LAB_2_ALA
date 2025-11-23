import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread



# Task 1
matrix = np.array([
    [2, 0, 0],
    [0, 3, 0],
    [0, 0, 4]
])

def eigen(matrix):

    eig_value, eig_vect = np.linalg.eig(matrix)

    eig_value = np.real(eig_value)

    eig_vect = np.real(eig_vect)

    print(f"Eigen value:{eig_value}")

    print(f"Eigen vector:{eig_vect}\n")

    perevirka  = True

    for i in range(len(eig_value)):

        col = eig_vect[:, i]

        res1 = matrix @ col

        print(f"Product matrix @ column \n {res1}")

        res2 = eig_value[i] * col

        print(f"Product lamda * col \n {res2}")

        print(np.allclose(res1, res2))
        if (np.allclose(res1, res2)) == True:
            continue
        else:
            perevirka = False

    return perevirka


# eigen(matrix)
