from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class LipschitzSGDScheduler(_LRScheduler):
	"""
	Schedule learning rate according to the inverse of the
	Lipschitz constant for SGD optimizers.
	
	Args:
		optimizer (optim.Optimizer): Wrapped optimizer
		model: The model, with a hook to the penultimate
			layer output
		classes: Number of classes
		bs: Batch size
	"""
	
	def __init__(self, optimizer, model, classes, bs):
		self.model = model
		self.classes = classes
		self.bs = bs
		super().__init__(optimizer)
	
	def get_lr(self):
		Kz = np.linalg.norm(self.model.output_hook.detach())
		k = self.classes
		return [(k-1) / (k * self.bs) * Kz]
