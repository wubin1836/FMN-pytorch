import cPickle as pickle

with open("./data/sample.pkl","rb") as pickle_file:
    data = pickle.load(pickle_file)

print(len(data))
print(data[0])