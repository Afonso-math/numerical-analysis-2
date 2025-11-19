import numpy as np

def hilbert_matrix(n):
    matrix = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = 1/(i + j + 1)
    return matrix

def norm_one(A):
    cols_sum = np.sum(np.abs(A), axis=0)
    return np.max(cols_sum)

def norm_infinite(A):
    max_row_sum = 0
    rows, cols = A.shape
    for i in range(rows):
        row_sum = 0
        for j in range(cols):
            row_sum += abs(A[i][j])
        if row_sum > max_row_sum:
            max_row_sum = row_sum
    return max_row_sum

def B_matrix(A):
    B = A.T * (norm_infinite(A) / norm_one(A))
    return B

def find_B(A, B_init, iteradas):
    rows, cols = A.shape
    I = np.eye(rows)
    B = B_init.copy()

    for i in range(iteradas):
        B = B @ (2 * I - A @ B)
    return B

def main():
    n = int(input("Insira a dimensão da matriz de Hilbert: "))
    iteradas = int(input("Insira o número de iteradas desejado: "))
    A = hilbert_matrix(n)
    B_inicial = B_matrix(A)
    B = find_B(A, B_inicial, iteradas)
    print(B)

main()