import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR

from snekhorn.dimension_reduction import SNEkhorn

class ECRTM(nn.Module):
    '''
        Effective Neural Topic Modeling with Embedding Clustering Regularization. ICML 2023

        Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu.
    '''
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0., pretrained_WE=None, embed_size=200,
                    cluster_distribution=None, cluster_mean=None, cluster_label=None, sinkhorn_alpha = 20.0,
                    beta_temp=0.2, weight_loss_ECR=250.0, alpha_ECR=20.0, sinkhorn_max_iter=1000, use_MOO = 1):
        super().__init__()

        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.use_MOO = use_MOO

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)
        self.theta_dropout = nn.Dropout(dropout)

        # self.mean_bn = nn.BatchNorm1d(num_topics)
        # self.mean_bn.weight.requires_grad = False
        # self.logvar_bn = nn.BatchNorm1d(num_topics)
        # self.logvar_bn.weight.requires_grad = False
        # self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        # self.decoder_bn.weight.requires_grad = False
        self.mean_bn = nn.BatchNorm1d(num_topics, affine = True)
        self.logvar_bn = nn.BatchNorm1d(num_topics, affine = True)
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)

        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))


        self.encoder1 = nn.Sequential(
            nn.Linear(vocab_size, en_units),
            nn.Softplus(),
            nn.Linear(en_units, en_units),
            nn.Softplus(),
            nn.Dropout(dropout)
        )
        
        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.ECR = ECR(weight_loss_ECR, alpha_ECR, sinkhorn_max_iter)

    # Same
    def get_beta(self):
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    # Same
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    # Same
    def encode(self, input):
        device = input.device
        e1 = self.encoder1(input)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))

        ##
        mu.requires_grad_(True)  # Ensure requires_grad=True
        logvar.requires_grad_(True)

        z = self.reparameterize(mu, logvar)
        #theta = F.softmax(z, dim=1).clone().detach().requires_grad_(True)
        theta = F.softmax(z, dim=1)
        theta.requires_grad_(True)

        ##
        perplexity = 30  # Adjust this value as needed
        output_dim = 40
        snekhorn = SNEkhorn(perp=perplexity, output_dim=output_dim, verbose=True)
        theta_reduced = snekhorn.fit_transform(theta.T.to(device))

        loss_KL = self.compute_loss_KL(mu, logvar)

        # return theta, loss_KL
        return theta_reduced.T, loss_KL


    # Same
    def get_theta(self, input):
        theta, loss_KL = self.encode(input)
        if self.training:
            return theta, loss_KL
        else:
            return theta

    # Same
    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    # Same
    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR

    # Same
    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def forward(self, indices, input, epoch_id=None):
        # input = input['data']
        bow = input[0]
        ##
        perplexity = 30  # Adjust this value as needed
        output_dim = 40
        snekhorn = SNEkhorn(perp=perplexity, output_dim=output_dim, verbose=True)
        bow_reduced = snekhorn.fit_transform(bow.T).T

        print(f"The dimensions of bow_reduced: {bow_reduced.shape}")

        theta_reduced, loss_KL = self.encode(bow_reduced)
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta_reduced, beta)), dim=-1)
        #recon = recon.T
        recon_loss = -(bow_reduced * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + loss_KL

        loss_ECR = self.get_loss_ECR()


        loss = loss_TM + loss_ECR

        rst_dict = {
            'loss_': loss,
            'loss_TM': loss_TM,
            'loss_ECR': loss_ECR,
            'theta_reduced': theta_reduced
        }

        return rst_dict

