import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Computes the supervised contrastive loss for a batch of feature embeddings and their corresponding labels.
        Args:
            features (torch.Tensor): A tensor of shape (B, D) containing the feature embeddings for a batch of B samples, each of dimension D.
            labels (torch.Tensor): A tensor of shape (B,) containing the class labels for each sample in the batch.
        Returns:
            torch.Tensor: A scalar tensor representing the computed supervised contrastive loss.
        Details:
            - Normalizes the feature embeddings to unit vectors.
            - Computes a pairwise cosine similarity matrix, scaled by the temperature parameter.
            - Constructs a mask to identify positive pairs (samples with the same label) and excludes self-comparisons.
            - Applies log-softmax to the similarity matrix.
            - Averages the log-probabilities over positive pairs for each sample.
            - Returns the mean negative log-probability as the loss.
        """
        
        device = features.device
        # normalize embeddings, so sim(z_i, z_j) = z_i ⋅ z_j
        features = F.normalize(features)
        # similarity matrix (cosine similarity) -> sim_ij = z_i ⋅ z_j / τ
        sim_matrix = torch.matmul(features, features.T) / self.temperature # shape: (B, B)

        # create label mask
        labels = labels.contiguous().view(-1, 1) # make sure this tensor is laid out correctly in memory, then reshape it from (B,) to (B, 1) -> column vector
        mask = torch.eq(labels, labels.T).float().to(device) # mask[i, j] = 1 if same class, else 0
        # -> torch.eq basically creates a matrix by comparing y_i and y_j, where we use labels.T to obtain a square matrix of shape (B, B)
        # remove self-comparisons
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0) # -> mask[i, i] = 0
        mask = mask * logits_mask

        # compute log-softmax over rows
        logits = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # average over positives
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum

        loss = -mean_log_prob_pos.mean()
        return loss