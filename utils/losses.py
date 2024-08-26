import torch
import torch.nn.functional as F

def compute_contrastive_loss(features, temperature=0.5):
    # Normalize the features
    features = F.normalize(features, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T)

    # Remove diagonal from similarity matrix and apply temperature scaling
    batch_size = features.size(0)
    mask = torch.eye(batch_size, dtype=torch.bool).to(features.device)
    similarity_matrix = similarity_matrix[~mask].view(batch_size, -1)
    similarity_matrix /= temperature

    # Create labels for contrastive loss
    labels = torch.arange(batch_size).to(features.device)
    labels = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    labels = labels[~mask].view(batch_size, -1)
    
    # Compute log-softmax of similarities
    logits = F.log_softmax(similarity_matrix, dim=1)

    # Compute loss
    loss = -torch.mean(torch.sum(labels * logits, dim=1))
    return loss
