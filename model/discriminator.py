import torch.nn as nn
import torch


class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)

		return x

class SpectralConvolution(nn.Module):
	def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias=True):
		super(SpectralConvolution, self).__init__()
		self.l = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

	def forward(self, x):
		return self.l(x)

class SpectralDiscriminator(nn.Module):
	def __init__(self, num_classes, ndf = 64):
		super(SpectralDiscriminator, self).__init__()

		self.conv1 = SpectralConvolution(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = SpectralConvolution(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = SpectralConvolution(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = SpectralConvolution(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = SpectralConvolution(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		# x = self.up_sample(x)
		# x = self.sigmoid(x)

		return x

class Hinge(nn.Module):
	def __init__(self, discriminator):
		super(Hinge, self).__init__()

		self.discriminator = discriminator

	def forward(self, fake_samples, real_samples=None, generator=True, label=None, new_hinge=False):
		if label is None:
			fake = self.discriminator(fake_samples)
		else:
			fake = self.discriminator(fake_samples, label)

		if generator:
			if new_hinge:
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