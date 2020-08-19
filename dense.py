import torch.nn as nn
import torchvision.models as models
class Dense(nn.Module):
	def __init__(self):
		super(Dense,self).__init__()
		net = models.densenet169(pretrained=False)
		features = net.features

		self.conv0 = features.conv0
		self.norm0 = features.norm0
		self.relu0 = features.relu0
		self.pool0 = features.pool0

		self.dense1 = features.denseblock1
		self.trans1 = features.transition1

		self.dense2 = features.denseblock2
		self.trans2 = features.transition2

		self.dense3 = features.denseblock3
		self.trans3 = features.transition3

		self.dense4 = features.denseblock4
		self.norm5 = features.norm5

	def forward(self, x):
		x = self.conv0(x)
		x = self.norm0(x)
		x = self.relu0(x)
		x1 = x = self.pool0(x)

		x = self.dense1(x)
		x2 = x = self.trans1(x)

		x = self.dense2(x)
		x3 = x = self.trans2(x)

		x = self.dense3(x)
		x4 = x = self.trans3(x)

		x = self.dense4(x)
		x5 = self.norm5(x)

		return x1,x2,x3,x4,x5

if __name__ == '__main__':
	import torch
	net = Dense()
	input = torch.randn(1,3,224,224)
	out1,out2,out3,out4,out5 = net(input)
	print(out1.size())
	print(out3.size())
	print(out5.size())



