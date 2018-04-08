import numpy as np
import pandas as pd
from read_file import load_json_file
from markov_chain import create_transition_matrix, m_i, p_i, pi_i, population_n, flow_d, usage_u, waiting_time_w, loss, create_probability_vector


if __name__ == '__main__':
    _lambda = 2
    m = 2
    c = 3
    k = 3

    m_is = [0] * (k + 1)

    for i in range(k+1):
        m_is[i] = m_i(i, c, m)
        print('M_%d = %.4f' % (i, m_is[i]))

    p_is = [0] * (k + 1)

    for i in range(k+1):
        p_is[i] = p_i(i, _lambda, c, m, previous_p_i=None if i == 0 else p_is[i - 1], current_m_i=m_is[i])
        print('P_%d = %.4f' % (i, p_is[i]))

    sum_of_p_is = sum(p_is)
    print('sum P = %.4f' % sum_of_p_is)

    pi_is = [0] * (k + 1)

    for i in range(k+1):
        pi_is[i] = pi_i(i, _lambda, c, m, sum_of_p_is, current_p_i=p_is[i])
        print('PI_%d = %.4f' % (i, pi_is[i]))

    population = population_n(k, pi_is)
    print('N = %.4f' % population)

    flow = flow_d(k, pi_is, m_is)
    print('D = %.4f' % flow)

    usage = usage_u(k, c, pi_is)
    print('U = %.4f' % usage)

    waiting_time = waiting_time_w(population, flow)
    print('W = %.4f' % waiting_time)

    l = loss(_lambda, pi_is)
    print('Loss = %.4f' % l)

    a = np.array([[-2.02, 2, 0, 0, 2], [2, -4.02, 4, 0, 0], [0, 2, -6.02, 6, 0], [0, 0, 2, -6.02, 0],
                  [0.02, 0.02, 0.02, 0.02, -2]])
    df = pd.DataFrame(a)
    df = df.swapaxes(1, 0, copy=False)
    print(df)
    print(create_probability_vector(df))
    # Expected result [2.0469, 2.0134, 1, 0.3322, 0.0539]
