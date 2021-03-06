import os
import h5py
import torch
import numpy as np
from networks import *
from PIL import Image
from torchvision import transforms

os.makedirs('embeddings_SupCon', exist_ok = True)

query_images = '/home/pulkit/Datasets/market1501/query/'
image_gallery = '/home/pulkit/Datasets/market1501/bounding_box_test/'
hf_gallery = h5py.File('embeddings_SupCon/gallery_embeddings.h5', 'w')
hf_query = h5py.File('embeddings_SupCon/query_embeddings.h5', 'w')

images_ = sorted(os.listdir(image_gallery)) # Contains the image names present in the gallery
queries_ = sorted(os.listdir(query_images)) # Contains the image names present in the query set

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = EmbeddingNet()
checkpoint = torch.load('checkpoints_market1501_SupCon/model_epoch_500.pth')
model_parameters = checkpoint['state_dict']
model.load_state_dict(model_parameters)
model.to(device)
model.eval()

totensor = transforms.ToTensor()

def extract_embeddings(dataset_path, dataset):

	print('Extracting Embeddings:')

	for batch_id, image_name in enumerate(dataset):
		
		img = Image.open(dataset_path + image_name).convert('RGB')
		img = totensor(img)
		img = img.unsqueeze(0)
		img = img.to(device)
		
		with torch.no_grad():
			embeddings = model.get_embeddings(img)
		
		embeddings = embeddings.cpu().numpy()
		
		if batch_id == 0:
			embeddings_array = embeddings
		else:
			embeddings_array = np.append(embeddings_array, embeddings, axis = 0)

	return embeddings_array

embeddings_query = extract_embeddings(query_images, queries_)
embeddings_gallery = extract_embeddings(image_gallery, images_)

hf_query.create_dataset('emb', data = embeddings_query)
hf_gallery.create_dataset('emb', data = embeddings_gallery)

hf_query.close()
hf_gallery.close()