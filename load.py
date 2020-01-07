import numpy as np
from sklearn.decomposition import PCA


sym=np.load('symmetry_function.npz')
pca=np.load('pca.npz')




#list of contents
print(sym.files)
#shape of symmetry function: number of data, number of atom, number of symmetry functions
print(sym['sym_func'].shape)

#derivative
#shape of derivative of symmetry function: number of data, number of atom, number of symmetry functions, number of atomsx3
print(sym['derivative'].shape)

#other contents
print(sym['elemental_composition'])


#other contents
print(sym['elements'])

#other contents
print(sym['feature_keys'])

#other contents
print(sym['tag'])

#list of contents for pca
print(pca.files)

#check the shape of transform matrix
print(pca['transform:Si'].shape)

#try obtain PCs for 0th data
print(sym['sym_func'][0].shape)

mean = pca['mean:Si']
print(mean.shape)
print(mean)
pcs=np.einsum('af,ft->at',sym['sym_func'][0]-mean, pca['transform:Si'])
#pcs=np.dot(sym['sym_func'][0]-pca['mean:Si'], pca['transform:Si'])
print(pcs.shape)

#1st PC
print('PCs for atom1')
print(pcs[0])
print(np.sum(pcs[0]))

#compare with original
print(sym['sym_func'][0][0])
#after subtract mean
print((sym['sym_func'][0]-mean)[0])

print(np.dot((sym['sym_func'][0]-mean)[0],pca['transform:Si'][0:,0]))
print(np.dot((sym['sym_func'][0]-mean)[0],pca['transform:Si'][0:,1]))
print("PC1")
print(pcs[0:,0])
print(np.sum(pcs[0:,0]))

print("PC2")
print(pcs[0:,1])
print(np.sum(pcs[0:,1]))

#matrix
print('vector for 1st PC')
print(pca['transform:Si'][0:,0])


print('vector for 2nd PC')
print(pca['transform:Si'][0:,1])


