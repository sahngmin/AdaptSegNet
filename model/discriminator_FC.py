import torch.nn as nn
import torch


class FCDiscriminator(nn.Module):

	def __init__(self):
		super(FCDiscriminator, self).__init__()

		self.fc1 = nn.Linear(128, 64)
		self.fc2 = nn.Linear(64, 32)
		self.fc3 = nn.Linear(32, 1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		x = self.relu(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)
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
				loss = torch.nn.ReLU()(torch.mean(real_samples) - torch.mean(fake))
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