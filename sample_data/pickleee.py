import pickle

pkl_file = open('filename.pkl', 'rb')
mydict2 = pickle.load(pkl_file)
pkl_file.close()

print(mydict2)
'''import pickle

a = [{'horizontal_flip': 'NO', 'L-CC': ['0_L_CC'], 'L-MLO': ['0_L_MLO'], 'R-MLO': ['0_R_MLO'], 'R-CC': ['0_R_CC']}]
with open('filename.pkl', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pkl', 'rb') as handle:
    b = pickle.load(handle)'''
