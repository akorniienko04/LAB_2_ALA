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



# Task 2


def pca(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

    image_sum = image.sum(axis = 2)
    # print(image_sum.shape)

    image_bw = image_sum / image_sum.max()

    plt.figure()
    plt.imshow(image_bw, cmap = 'gray')
    plt.show()

    x = image_bw.astype(float)

    x_centered = x - np.mean(x, axis = 0)

    cov_matrix = np.cov(x_centered, rowvar = False)

    eigval, eigvect = np.linalg.eig(cov_matrix)

    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvect = eigvect[:, idx]

    explained_variance = eigval / np.sum(eigval)

    cmlt_var = np.cumsum(explained_variance)

    k = np.argmax(cmlt_var >= 0.95) + 1
    print("Number of comp needed to cover 95% variance: ", k)

    plt.figure()
    plt.plot(cmlt_var)
    plt.axhline(0.95, color = "black", linestyle = '--')
    plt.axvline(k - 1, color = "black", linestyle = '--')
    plt.show()

    w = eigvect[:, :k]

    z = x_centered @ w

    x_rec = z @ w.T

    x_rec = x_rec + np.mean(x, axis = 0)

    plt.figure(figsize = (6, 6))
    plt.imshow(x_rec, cmap = 'gray')
    plt.title(f"{k}: 95% var")
    plt.axis('off')
    plt.show()

    k_val = [5, 10, k, 50, 100, 200]

    plt.figure(figsize = (15, 10))

    for i, k_for_test in enumerate(k_val, 1):
        eig_vect_test = eigvect[:, :k_for_test]
        pca_proj = x_centered @ eig_vect_test
        x_rec_k = pca_proj @ eig_vect_test.T + np.mean(x, axis=0)

        plt.subplot(2, 3, i)
        plt.imshow(x_rec_k, cmap = 'gray')
        plt.title(f"k = {k_for_test}")
        plt.axis('off')

    plt.show()

    return 0


image_raw = imread("/Users/artemkorniienko/R_for_DATA/tiger.jpg")
# print(image_raw.shape)

# pca(image_raw)


#Task 3

def encrypt_message(message, key_matrix):

    message_vector = np.array([ord(char) for char in message])

    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)

    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))

    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)

    return encrypted_vector


def decrypt_message(encrypted_vector, key_matrix):

    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)

    d_inv = np.diag(1 / eigenvalues)

    p_inv = np.linalg.inv(eigenvectors)

    a_inv = eigenvectors @ d_inv @ p_inv

    message_vector = a_inv @ encrypted_vector

    message = "".join([chr(int(round(x))) for x in message_vector])

    return message


message = "Hello, World!"

key_matrix = np.random.randint(0, 256, (len(message), len(message)))

print("Orig message:", message)

encrypted = encrypt_message(message, key_matrix)
print("Encrypted message:", encrypted)

decrypted = decrypt_message(encrypted, key_matrix)
print("Decrypted message:", decrypted)



