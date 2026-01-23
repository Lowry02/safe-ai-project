import torch
import torch.nn as nn
import torch.nn.functional as F

class SupCon(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupCon, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: Tensor of shape (B, D), where B is the batch size and D is the size of the embeddings?
        labels:   Tensor of shape (B,)
        """
        # ---------- :( INUTILE ----------
        # cos_sim = nn.CosineSimilarity()
        # for i, feature in enumerate(features):
        #     label_i = labels[i]
        #     P_i = [idx for j, idx in enumerate(labels) if label_i == idx and j != i] # this should be the list of elements in the batch that have the same label
        #     # TODO: remove label_i -> remove element == i

            
        #     logarithm = 0
        #     for p in P_i:
        #         numerator = torch.exp(cos_sim(feature, features[p]) / self.temperature)
        #         denominator = 0
        #         for a in range(len(features)): # TODO: remove i
        #             denominator += torch.exp(cos_sim(feature, features[a]) / self.temperature)
        #         logarithm += torch.log(numerator / denominator)

        #     result = logarithm / len(P_i)
        #     return -result

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
    

class CombinedLoss(nn.Module):
    def __init__(self, alpha = 0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.entropy_loss = nn.CrossEntropyLoss()
        self.supcon_loss = SupCon()

    def forward(self, embeddings, outputs, labels):
        return self.entropy_loss(outputs, labels) * (1 - self.alpha) + self.supcon_loss(embeddings, labels) * self.alpha