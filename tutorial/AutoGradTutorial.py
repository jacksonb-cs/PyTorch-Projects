import torch

# In order to compute the gradients required for back propagation, we can use
# torch.autograd to automatically compute the gradient for any computational
# graph. The tutorial refers to autograd as a built-in differentiation engine.

input_size = 5
output_size = 3

x = torch.ones(input_size)   # Input tensor
y = torch.zeros(output_size)  # Expected output

# This is basically a one layer neural network (looks like a linear layer to me)
w = torch.randn(input_size, output_size, requires_grad=True)
b = torch.randn(output_size, requires_grad=True)

z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(w.grad)
loss.backward()

# This is possible because autograd keeps a record of data (tensors) and all
# executed operations in a directed acyclic graph consisting of Function
# objects.
# In this DAG, leaves are input tensors and roots are output tensors. By tracing
# this graph from roots to leaves, you can compute the gradients using the
# chain rule!
print(w.grad)
print(b.grad)

# In a forward pass, autograd simultaneously does two things:
#	- Run the requested operation
#	- Maintain the operation's gradient function in the DAG
# The backward pass begins when .backward() is called on the DAG root.
# Autograd then does the following:
#	- Computes the gradients from each .grad_fn
#	- Accumulates them in the respective (output?) tensor's .grad attribute
#	- Using the chain rule, propagates all the way to the leaf tensors

# IMPORTANT: DAGs are dynamic in PyTorch. After each .backward() call,
# autograd starts populating a new graph. In other words, the graph is
# recreated from scratch. This is what allows one to use control flow statements
# in their model. This means you can change the shape, size, and operations at
# every iteration if necessary.

print(f'Before no_grad(), requires_grad: {z.requires_grad}')

with torch.no_grad():
	z = torch.matmul(x, w) + b
print(f'After no_grad(), requires_grad: {z.requires_grad}')

# Same thing with detach()
z_det = z.detach()
print(f'Detached version, requires_grad: {z_det.requires_grad}')

# The above disables gradient tracking. This is useful if you want to:
#	- Mark some parameters as 'frozen' parameters
#		= Common to do when finetuning a pretrained network
#	- Speed up computations when you are only doing forward passes
#		= For example, in a production environment

# Bonus stuff: Tensor Gradients and Jacobian Products
# TLDR: Sometimes output is an arbitrary tensor and we have a scalar loss
# function. We want to compute the gradient with respect to some parameters,
# but we can't for some reason (due to the aforementioned tensor/function).
# To work around this, we can calculate a so-called Jacobian product instead of
# the actual gradient.

# Note: We aren't computing the Jacobian matrix itself, only the Jacobian
# product v.transpose() * J for a given input vector v. We do this by calling
# backward() with v as an argument. The size of v should be the same as the
# original tensor with respect to which we want to compute the product.

inp = torch.eye(5, requires_grad=True)	# This is the identity matrix lol
out = (inp + 1).pow(2)

out.backward(torch.ones_like(inp), retain_graph=True)
print(f'First call\n{inp.grad}\n')

out.backward(torch.ones_like(inp), retain_graph=True)
print(f'Second call\n{inp.grad}\n')

inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f'Call after zeroing gradients\n{inp.grad}\n')

# Note the above ^^^
# The second time calling backward with the same argument changes the gradient
# again because PyTorch accumulates the gradients, i.e. the value of computed
# gradients is added to the grad property of all leaf nodes of the computational
# graph. In order to compute the proper gradients, you have to zero out the grad
# property beforehand. IRL training, an optimizer helps use do this.

# Note: Calling backward() (w/o parameters) is essentially equivalent to calling
# backward(torch.tensor(1.0))
# This is a useful way to compute the gradients in case of a scalar-valued
# function, such as loss during neural network training.

print('\n========== DONE ==========')