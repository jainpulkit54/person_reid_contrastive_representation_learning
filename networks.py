import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EmbeddingNet(nn.Module):

	def __init__(self):
		super(EmbeddingNet, self).__init__()
		resnet50 = models.resnet50(pretrained = False, progress = True)
		resnet50.fc = nn.Sequential(
			nn.Linear(2048, 2048),
			nn.ReLU(inplace = True),
			nn.Linear(2048, 128)
			)
		resnet50 = list(resnet50.children())
		self.encoder = nn.Sequential(*resnet50[:-1])
		self.head = nn.Sequential(*resnet50[-1])

	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.shape[0], -1)
		x = self.head(x)
		x = F.normalize(x, p = 2, dim = 1)
		return x

	def get_embeddings(self, x):
		return self.forward(x)

	def get_embeddings_for_classifier(self, x):
		x = self.encoder(x)
		x = x.view(x.shape[0], -1)
		return x

class ClassifierNet(nn.Module):

	def __init__(self):
		super(ClassifierNet, self).__init__()
		self.fc_layers = nn.Linear(2048, 10)

	def forward(self, x):
		return self.fc_layers(x)