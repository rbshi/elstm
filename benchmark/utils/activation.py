import torch
import math 
import numpy as np  
import torch.nn.functional as F
 
def hard_sigmoid(x):
	x = x
	x = F.threshold(-x, -2, -0.5)
	x = F.threshold(-x, -2, -0.5)
	return x 