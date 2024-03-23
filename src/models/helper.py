from torch import nn

class tupleSequential(nn.Sequential):
	def forward(self, *inputs):
		for module in self._modules.values():
			inputs = module(*inputs)
		return inputs
