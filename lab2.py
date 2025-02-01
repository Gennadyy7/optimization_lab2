from typing import Annotated

import numpy as np

from lab1 import lab1

# Условие
cT = np.array([1, 1, 0, 0, 0])

A = np.array([
    [-1,  1,  1,  0,  0],
    [ 1,  0,  0,  1,  0],
    [ 0,  1,  0,  0,  1],
])

xT = np.array([0, 0, 1, 3, 2])

B = [3, 4, 5]


def lab2(cT, A, xT, B: [Annotated[int, 'Индексация с единицы']]):
    A_B_prev = None
    A_B_inv = None
    k = None
    while True:
        A_B = A[:, [j-1 for j in B]]
        if k:
            differing_columns = np.any(A_B != A_B_prev, axis=0)
            num_differing_columns = np.sum(differing_columns)
            if num_differing_columns == 1:
                A_B_inv = lab1(A_B_inv, A_B[:, k-1], k)
            else:
                A_B_inv = np.linalg.inv(A_B)
        else:
            A_B_inv = np.linalg.inv(A_B)
        A_B_prev = A_B # Вне алгоритма
        cT_B = [cT[j-1] for j in B]
        uT = np.dot(cT_B, A_B_inv)
        ΔT = np.dot(uT, A) - cT
        if np.all(ΔT >= 0):
            return xT
        Δ1, j0 = next((component, index) for index, component in enumerate(ΔT, start=1)
                      if component < 0)
        z = np.dot(A_B_inv, A[:, j0-1])
        θT = np.array([θ(z_i, xT[B[index-1]-1]) for index, z_i in enumerate(z, start=1)])
        θ_0, k = np.min(θT), np.argmin(θT) + 1
        if np.isinf(θ_0):
            raise Exception(
                'Целевой функционал задачи не ограничен сверху на множестве допустимых планов'
            )
        j_star = B[k-1]
        B[k-1] = j0
        xT[j0-1] = θ_0
        for i in range(1, len(B)+1):
            if i != k:
                xT[B[i-1]-1] -= θ_0 * z[i-1]
        xT[j_star-1] = 0

def θ(z_i, x_j_i):
    return x_j_i/z_i if z_i > 0 else float('inf')

if __name__ == '__main__':
    try:
        print(lab2(cT, A, xT, B))
    except Exception as e:
        print(e)