from pandas import DataFrame


def preprocess(markov_chain):
    """

    :param markov_chain:
    :type: markov_chain: dict
    :return:
    :rtype: dict
    """
    data = {}

    keys = list(markov_chain.keys())

    i = 0
    for key in keys:
        d = {}

        for value in markov_chain[key]:
            target = value['target']
            d[target] = value['rate']

        row_sum = sum(d.values())
        row = [0]*len(keys)
        row[i] = row_sum*(-1)

        for target, rate in d.items():
            index = keys.index(target)
            row[index] = rate

        data[key] = row

        i += 1

    return data


def create_df(preprocessed_markov_chain):
    """

    :param preprocessed_markov_chain:
    :type preprocessed_markov_chain: dict
    :return:
    :rtype: DataFrame
    """
    df = DataFrame(preprocessed_markov_chain, index=preprocessed_markov_chain.keys())
    df = df.swapaxes(1, 0, copy=False)

    return df
