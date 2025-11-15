import torch
import torch.nn.functional as F
from torch.autograd import grad


def entropy_loss(logits):
    """
    Calculate the entropy of the probability distribution created by the logits.
    The entropy is maximized when the distribution is uniform across all classes.
    """
    # Apply softmax to convert logits to probabilities
    probs = F.softmax(logits, dim=1)
    
    # Compute the mean entropy of the probabilities
    entropy = torch.mean(torch.sum(probs * torch.log(probs), dim=1))

    return entropy


def compute_gradient_penalty(output, input_data):    
    """
    Compute the gradient penalty. This works as a regularizer.
    """
    # Calculate gradients of output w.r.t input data
    gradients = grad(outputs=output, inputs=input_data,
                     grad_outputs=torch.ones_like(output),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    # Compute the L2 norm of the gradients
    gradient_penalty = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2, 3]) + 1e-12)
    
    # Return the mean of the gradient penalty
    return gradient_penalty.mean()


