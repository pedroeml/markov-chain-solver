from read_file import load_json_file
from preprocess import preprocess, create_df

markov_chain = load_json_file('markov_chain_example.json')
data = preprocess(markov_chain)
df = create_df(data)

print(df)
