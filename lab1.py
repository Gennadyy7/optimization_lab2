from typing import Annotated

import numpy as np

# Условие 1
A = np.array([
    [1, -1,  0],
    [0,  1,  0],
    [0,  0,  1],
])

A_inv = np.array([
    [1,  1,  0],
    [0,  1,  0],
    [0,  0,  1],
])

x = np.array([
    1,
    0,
    1,
])

i = 3

# Условие 2
A_inv = np.array([
    [-24,  20,  -5],
    [18,  -15,  4],
    [5,  -4,  1],
])

x = np.array([
    2,
    2,
    2,
])

i = 2

# Функция, реализующая метод
def lab1(A_inv, x, i: Annotated[int, 'Индексация с единицы'], optimized_dot=True):
    l = np.dot(A_inv, x)
    i_component = l[i-1]
    if not i_component:
        raise Exception('Реализовать метод невозможно (Â не обратима)')
    l[i-1] = -1
    l_capped = (-1 / i_component) * l
    E_n = np.eye(A_inv.shape[0])
    Q = np.hstack((E_n[:, :i-1], l_capped[:, np.newaxis], E_n[:, i:]))
    if optimized_dot:
        return optimized_Q_Ainv_dot(Q, A_inv, i)
    else:
        return np.dot(Q, A_inv)

def optimized_Q_Ainv_dot(Q, A_inv, i: Annotated[int, 'Индексация с единицы']):
    A_capped_inv = np.zeros(A_inv.shape)
    for col_index, column in enumerate(A_inv.T):
        for row_index, row in enumerate(Q):
            A_capped_inv[:, col_index][row_index] = (
                    Q[row_index, :][row_index] * column[row_index] + Q[row_index, :][i - 1] * column[i - 1]
            )
            if row_index == i-1:
                A_capped_inv[:, col_index][row_index] /= 2
    return A_capped_inv


if __name__ == '__main__':
    try:
        print(lab1(A_inv, x, i))
    except Exception as e:
        print(e)