import numpy as np

def hilbert_matrix(n):
    matrix = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = 1 / (i + j + 1)
    return matrix

def norm_one(A):
    return np.max(np.sum(np.abs(A), axis=0))

def norm_infinite(A):
    return np.max(np.sum(np.abs(A), axis=1))

def B_matrix_Newton(A):
    return A.T * (norm_infinite(A) / norm_one(A))

def B_matrix_Neumann(A):
    # Usada na série de Neumann
    alfa = 10**(-16) + norm_infinite(A) * norm_infinite(A.T)
    return A.T / alfa

def newton_inverse(A, B_init, iteradas):
    rows, cols = A.shape
    I = np.eye(rows)
    X = B_init.copy()

    for i in range(iteradas):
        X = X @ (2 * I - A @ X)
    return X

def compute_inverse_series(A, B_init, k):
    rows, cols = A.shape
    I = np.eye(rows)
    A_inv = np.zeros_like(A)

    for j in range(k + 1):
        A_inv += B_init @ np.linalg.matrix_power(I - A @ B_init, j)
    return A_inv

def erro(A_inv, A):
    E = A_inv @ A - np.eye(A.shape[0])
    return np.max(np.sum(np.abs(E), axis=1))

def main():
    n = int(input("Insira a dimensão da matriz de Hilbert: "))
    iteradas = int(input("Insira o número de iteradas desejado: "))

    A = hilbert_matrix(n)

    # Métodos
    B_newton = B_matrix_Newton(A)
    B_neumann = B_matrix_Neumann(A)

    A_inv_newton = newton_inverse(A, B_newton, iteradas)
    A_inv_neumann = compute_inverse_series(A, B_neumann, iteradas)
    A_inv_LU = np.linalg.inv(A)

    # Erros
    erro_newton = erro(A_inv_newton, A)
    erro_neumann = erro(A_inv_neumann, A)
    erro_LU = erro(A_inv_LU, A)

    # Resultados
    print("\nMatriz de Hilbert:")
    print(A)

    print("\nInversa (Método de iterações):")
    print(A_inv_newton)

    print("\nInversa (Série de Neumann):")
    print(A_inv_neumann)

    print("\nInversa (np.linalg.inv - LU):")
    print(A_inv_LU)

    print("\n--- ERROS FUNCIONAIS ---")
    print(f"Erro (Newton): {erro_newton}")
    print(f"Erro (Série de Neumann): {erro_neumann}")
    print(f"Erro (LU): {erro_LU}")


main()
