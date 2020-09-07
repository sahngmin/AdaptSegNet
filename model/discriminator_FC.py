import torch.nn as nn
import torch


class FCDiscriminator(nn.Module):

	def __init__(self, output=2):
		super(FCDiscriminator, self).__init__()

		# if gan == 'Hinge':
		# 	output=10
		# else:
		# 	output =2
		self.fc1 = nn.Linear(50*4*4, 100)
		self.bn1 = nn.BatchNorm1d(100)
		self.fc2 = nn.Linear(100, 2)
		self.log_softmax = nn.LogSoftmax(dim=1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.fc2(x)
		# x = self.log_softmax(x)

		#x = self.up_sample(x)
		# x = self.sigmoid(x)
		return x



class FCDiscriminator_Spec(nn.Module):

	def __init__(self):
		super(FCDiscriminator_Spec, self).__init__()

		self.fc1 = nn.utils.spectral_norm(nn.Linear(50*4*4, 100))
		self.bn1 = nn.utils.spectral_norm(nn.BatchNorm1d(100))
		self.fc2 = nn.utils.spectral_norm(nn.Linear(100, 2))
		self.log_softmax = nn.LogSoftmax(dim=1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.fc2(x)
		# x = self.log_softmax(x)

		#x = self.up_sample(x)
		# x = self.sigmoid(x)
		return x




class Hinge(nn.Module):
	def __init__(self, discriminator):
		super(Hinge, self).__init__()

		self.discriminator = discriminator

	def forward(self, fake_samples, real_samples=None, generator=True, label=None, new_Hinge=False):
		if label is None:
			fake = self.discriminator(fake_samples)
		else:
			fake = self.discriminator(fake_samples, label)

		if generator:
			if new_Hinge:
				loss = torch.nn.ReLU()(torch.mean(self.discriminator(real_samples)) - torch.mean(fake))
			else:
				loss = -torch.mean(fake)
		else:
			if label is None:
				real = self.discriminator(real_samples)
			else:
				real = self.discriminator(real_samples, label)

			loss = (torch.mean(torch.nn.ReLU()(1 - real)) +
					torch.mean(torch.nn.ReLU()(1 + fake)))
		return loss