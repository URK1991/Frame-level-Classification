import torch
import torch.nn.functional as nnf

def compute_ce_loss(output, target, criterion):
    """Compute the cross-entropy loss."""
    return criterion(output, target)

def compute_re_loss(prob, target):
    """Compute the rejection error loss based on predictions."""
    tm = []
    for i in range(len(prob)):
        mx = torch.max(prob[i])
        ind = int(torch.argmax(prob[i]))
        t_ind = int(torch.argmax(target[i]))

        # Determine rejection error based on prediction and target indices
        re = 1 if t_ind != ind else 1 - mx
        tm.append(re)

    return sum(tm) / len(tm) if tm else 0  # Prevent division by zero

def my_custom_loss(output, target, criterion):
    """Calculate the custom loss combining CE loss and rejection error loss."""
    # Compute cross-entropy loss
    ce_loss = compute_ce_loss(output, target, criterion)
    
    # Compute softmax probabilities
    prob = nnf.softmax(output, dim=1)
    
    # Compute rejection loss
    re_loss = compute_re_loss(prob, target)
    
    # Combine losses 
    #equal distribution of the two losses
    t_loss = 0.5 * ce_loss + 0.5 * re_loss
    return t_loss
