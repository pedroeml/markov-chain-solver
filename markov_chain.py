import numpy as np
from preprocess import preprocess, create_df


def create_transition_matrix(markov_chain):
    """

    :param markov_chain:
    :type: markov_chain: dict
    :return:
    :rtype: DataFrame
    """
    data = preprocess(markov_chain)
    df = create_df(data)

    return df


def m_i(i, c, m):
    if i < 0:
        raise IndexError
    return min(i, c) * m


def p_i(i, _lambda, c, m, previous_p_i=None, current_m_i=None):
    if i == 0:
        return 1
    elif i < 0:
        raise IndexError
    elif previous_p_i is not None:
        if current_m_i is not None:
            return previous_p_i * _lambda / current_m_i
        else:
            return previous_p_i * _lambda/ m_i(i, c, m)
    elif current_m_i is not None:
            return p_i(i - 1, _lambda, c, m) * _lambda / current_m_i

    return p_i(i - 1, _lambda, c, m) * _lambda / m_i(i, c, m)


def pi_i(i, _lambda, c, m, sum_p, current_p_i=None):
    if current_p_i is not None:
        return current_p_i/sum_p

    return p_i(i, _lambda, c, m) / sum_p


def n_i(i, current_pi_i):
    return current_pi_i * i


def population_n(k, pi_is):
    n_is = [0] * k

    for i in range(1, k+1):
        n_is[i-1] = n_i(i, pi_is[i])

    return sum(n_is)


def d_i(currrent_pi_i, current_m_i):
    return currrent_pi_i * current_m_i


def flow_d(k, pi_is, m_is):
    d_is = [0] * k

    for i in range(1, k+1):
        d_is[i-1] = d_i(pi_is[i], m_is[i])

    return sum(d_is)


def u_i(i, c, current_pi_i):
    return current_pi_i * min(i, c)/c


def usage_u(k, c, pi_is):
    u_is = [0] * k

    for i in range(1, k+1):
        u_is[i-1] = u_i(i, c, pi_is[i])

    return sum(u_is)


def waiting_time_w(population, flow):
    return population/flow


def loss(_lambda, pi_is):
    return pi_is[-1] * _lambda


def create_probability_vector(transition_matrix):
    a = np.zeros((len(transition_matrix.columns), len(transition_matrix.columns)))

    i = 0
    max_zeros = (i, 0)  # index, count

    # Finds the equation which has more 0 coeficients than others
    for column in transition_matrix.columns:
        a[i] = transition_matrix[column].values
        count_zeros = a[i].size - np.count_nonzero(a[i])

        if count_zeros > max_zeros[1]:
            max_zeros = (i, count_zeros)

        i += 1

    # From the equation found, store the first positive nonzero coeficient and replace it with a 0 in 'a' ndarray
    nonzero_indexes = (a[max_zeros[0]] > 0).nonzero()  # the indexes of the elements which aren't negative or zero
    first_nonzero_index = nonzero_indexes[0][0]
    first_nonzero_element = a[max_zeros[0]][first_nonzero_index]
    a[max_zeros[0]][first_nonzero_index] = 0.  # replaces the first nonzero element by
    b = np.zeros(len(transition_matrix.columns))
    b[max_zeros[0]] = first_nonzero_element * (-1)  # multiply by -1 because it's switching from 'a' to 'b'
    x = np.linalg.solve(a, b)

    return x
