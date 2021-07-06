import os
import argparse
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser('Arguments for tsne embeddings settings')
parser.add_argument('--dataset_path', type = str, default = '/home/pulkit/Desktop/MTP/person_reid/market1501_test', help = 'The dataset path')
parser.add_argument('--embeddings_folder', type = str, default = 'market1501_test_set_embeddings_SupCon', help = 'The path where the embeddings are stored')
parser.add_argument('--image_file_name', type = str, default = 'market1501_test_set_embeddings_SupCon_', help = 'The image file')
args = parser.parse_args()

classes = sorted(os.listdir(args.dataset_path))
clss = []
for cls_num in classes:
	clss.append(str(int(cls_num)))

class_num = np.load(args.embeddings_folder + '/images_class.npy')
image_embeddings = np.load(args.embeddings_folder + '/images_embeddings.npy')

print('Files successfully loaded')
print(image_embeddings.shape)
print(class_num.shape)

tsne = TSNE(n_components = 3)
x = tsne.fit_transform(image_embeddings)
np.random.seed(0)

def plot_embeddings_3d(embeddings, targets):

	fig = plt.figure()
	ax = Axes3D(fig)
	colors = np.random.rand(len(clss), 3)

	legend = []

	for i, class_num in enumerate(clss):

		if i >= 100 and i <= 200:

			legend.append(class_num)
			inds = np.where(targets == int(class_num))[0]
			x = embeddings[inds, 0]
			y = embeddings[inds, 1]
			z = embeddings[inds, 2]
			ax.scatter(x, y, z, alpha = 1, color = colors[i, :])

	plt.legend(legend)
	plt.savefig(args.image_file_name + 'tsne3d.png')
	plt.show()

def plot_embeddings_2d(embeddings, targets):

	fig, ax = plt.subplots()
	colors = np.random.rand(len(clss), 3)

	legend = []

	for i, class_num in enumerate(clss):

		if i >= 100 and i <= 200:

			legend.append(class_num)
			inds = np.where(targets == int(class_num))[0]
			x = embeddings[inds, 0]
			y = embeddings[inds, 1]
			ax.scatter(x, y, alpha = 1, color = colors[i, :])

	plt.legend(legend)
	plt.savefig(args.image_file_name + 'tsne2d.png')
	plt.show()

plot_embeddings_3d(x, class_num)
# plot_embeddings_2d(x, class_num)