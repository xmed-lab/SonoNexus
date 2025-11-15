import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import math

class Codebook(nn.Module):
    def __init__(self, latent_dim, num_codebook_vectors):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        print("codebooks: {} with beta 0.25".format(self.num_codebook_vectors))
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.num_codebook_vectors, latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

        self.embedding_mask = nn.Embedding(self.num_codebook_vectors, latent_dim)
        self.embedding_mask.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

        self.pre_conv = nn.Conv2d(latent_dim, latent_dim, 1)
        #self.post_conv = nn.Conv2d(latent_dim, latent_dim, 1)

        self.beta = 0.25

    def forward(self, z, mask):
        z = self.pre_conv(z)
        mask = mask.permute(0, 2, 3, 1)

        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)
        
        embedding_weight = self.embedding.weight
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, embedding_weight.t()))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, embedding_weight).view(z.shape)


        embedding_weight_mask = self.embedding_mask.weight
        d_mask = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight_mask**2, dim=1) - \
            2*(torch.matmul(z_flattened, embedding_weight_mask.t()))
        min_encoding_indices_mask = torch.argmin(d_mask, dim=1)
        z_q_mask = F.embedding(min_encoding_indices_mask, embedding_weight_mask).view(z.shape)
        

        z_q = z_q * (1-mask) + z_q_mask * mask

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)

        #z_q = self.post_conv(z_q)

        return z_q, loss
