import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# PCA for Landsat9 SWIR scenes 
# does DOS1 atmospherics pre processing

# collapses the first n-1 dimensions of arr, PCA's, then rewrites the last dimension
# with the pca coefficients and returns the rewritten array plus the principal components
# and their singular values
def get_pca(arr):
	pca_mat = arr.reshape((np.prod(arr.shape[:-1]), arr.shape[-1]))
	#plt.scatter(pca_mat[:,0], pca_mat[:,1])
	#plt.show()
	U, S, Vh = np.linalg.svd(pca_mat, full_matrices=False)
	#print(Vh)
	V = Vh.T 
	proj_mat = np.dot(arr, V)#/ np.linalg.norm(arr, axis=-1, keepdims=True)
	return proj_mat, Vh, S

def load_img(img_path):
	return np.array(Image.open(img_path))

def findDNmin(inputRaster):
    uniques, counts = np.unique(inputRaster, return_counts=True)
    sumTot = np.sum(inputRaster)*0.0001
    sumPartial=0
    index=0
    #uniques values are already sorted
    for value in uniques:
        sumPartial += value*counts[index]    
        if sumPartial>=sumTot:
            DNm = value
            break
        index+=1 
    return DNm

def preprocess(img):
	DNmin = findDNmin(img)
	return img - (DNmin - 100)


def pstats(arr):
	print(f'min: {np.min(arr)}, max: {np.max(arr)}, shape: {np.shape(arr)}, dtype: {arr.dtype}')

if __name__ == '__main__':
	#arr = np.array(list(range(4*4*3))).reshape((4,4,3)) #W, H, C
	b6 = np.expand_dims(load_img('b6.tiff'), 2)
	b7 = np.expand_dims(load_img('b7.tiff'), 2)

	imgs = (b6, b7)
	imgs = tuple(preprocess(img) for img in imgs)

	#b6 = b6/np.max(b6)
	#b7 = b7/np.max(b7)
	arr = np.concatenate(imgs, axis=2)
	print(arr.shape)

	pca_image, pcomps, S = get_pca(arr)
	bw = pca_image[:,:,0] *-1
	print('components:', pcomps)
	print('singular values:', S)

	for img in imgs:
		pstats(img)
	#	plt.imshow(img, cmap=plt.get_cmap('gray'))
	#	plt.show()

	pstats(bw)
	plt.imshow(bw, cmap=plt.get_cmap('gray'))
	plt.savefig('out.tiff')
	plt.show()
