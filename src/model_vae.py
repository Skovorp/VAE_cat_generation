import torch
from torch import nn

from src.utils import ClampLayer
from src.utils import kl, gaussian_log_pdf
from itertools import chain
from src.utils import print_stats


class BaseVAE(nn.Module):
    def __init__(self, d, D, sigma_loss):
        """
        Initialize model weights.
        Input: d, int - the dimensionality of the latent space.
        Input: D, tuple(int) - the dimensionality of the latent space.
        """
        super().__init__()
        self.d = d
        self.D = D
        self.sigma_loss = torch.tensor([sigma_loss])

    def proposal_distr(self, x):
        """
        Generate proposal distribution over z.
        Note that sigma is positive by design of neural network.
        Input: x, Tensor of shape n x D.
        Return: tuple(Tensor, Tensor),
                Each Tensor is a matrix of shape n x d.
                The first one is mu, the second one is sigma.
        """
        raise NotImplementedError()

    def prior_distr(self, x):
        """
        Generate prior distribution over z.
        Note that sigma is positive by design of neural network.
        Input: x, Tensor of shape n x D.
        Return: tuple(Tensor, Tensor),
                Each Tensor is a matrix of shape n x d.
                The first one is mu, the second one is sigma.
        """
        n = x.shape[0]
        mu = torch.zeros(n, self.d, device=x.device)
        sigma = torch.ones(n, self.d, device=x.device)
        return mu, sigma

    def sample_latent(self, mu, sigma, K=1):
        """
        Generate samples from Gaussians with diagonal covariance matrices in latent space.
        Samples must be differentiable w. r. t. parameters of distribution!
        Use reparametrization trick.
        Input: mu, Tensor of shape n x d - mean vectors for n Gaussians.
        Input: sigma, Tensor of shape n x d - standard deviation vectors
               for n Gaussians.
        Input: K, int - number of samples from each Gaussian.
        Return: Tensor of shape n x K x d.
        """
        n = mu.shape[0]
        noise = torch.normal(0, 1, (n, K, self.d), device=mu.device)
        mu = torch.unsqueeze(mu, 1)
        sigma = torch.unsqueeze(sigma, 1)
        return mu + noise * sigma

    def generative_distr(self, z):
        """
        Compute a tensor of parameters of Bernoulli distribution over x
        given a tensor of latent representations.
        Input: z, Tensor of shape n x K x d - tensor of latent representations.
        Return: Tensor of shape n x K x D - parameters of Bernoulli distribution.
        """
        raise NotImplementedError()
    
    def reconstruction_log_likelihood(self, x_true, x_distr):
        """
        Compute some log-likelihood of x_true judging by x_distr
        Input: x_true, Tensor of shape n x D1 x D2 ... Dn
        Input: x_distr, Tensor of shape n x K x D1 x D2 ... Dn
        Return: Tensor of shape n x K - log-likelihood for each pair of an object
                and a corresponding distribution."""
        x_true = torch.unsqueeze(x_true, 1) # n k ds
        sigma = self.sigma_loss.to(x_true.device)
        
        res = -1 * (x_true - x_distr) ** 2 / (2 * sigma ** 2)
        res = res - torch.log(sigma) - 0.5 * torch.log(torch.tensor(2 * torch.pi))
        return res.sum(axis=list(range(2, len(x_true.shape))))
        # return gaussian_log_pdf(x_distr, self.sigma_loss.to(x_true.device), x_true)
        
        
    def batch_vlb(self, batch, K=1):
        """
        Compute VLB for batch. The VLB for batch is an average of VLBs for batch's objects.
        VLB must be differentiable w. r. t. model parameters, so use reparametrization!
        Input: batch, Tensor of shape n x D.
        Return: Tensor, scalar - VLB.
        """
        mu, sigma = self.proposal_distr(batch) # each (n, d)
        mu_prior, sigma_prior = self.prior_distr(batch) # each (n, d)
        z_samples = self.sample_latent(mu, sigma, K) # n, k, d
        x_samples = self.generative_distr(z_samples)
        recon = self.reconstruction_log_likelihood(batch, x_samples) # (n, K)
        assert len(recon.shape) == 2
        kl_res = kl(mu, sigma, mu_prior, sigma_prior) # (n) summed over k
        assert len(kl_res.shape) == 1

        recon = recon.mean(axis=1) # (n)
        #dont need to take mean over k
        
        # print('recon.sum():', recon.sum())
        # print('kl_res.sum():', kl_res.sum())

        return (recon.sum() - kl_res.sum()) / batch.shape[0]

    def generate_samples(self, num_samples):
        """
        Generate samples from the model.
        Tip: for visual quality you may return the parameters of Bernoulli distribution instead
        of samples from it.
        Input: num_samples, int - number of samples to generate.
        Return: Tensor of shape num_samples x D.
        """
        mu_prior, sigma_prior = self.prior_distr(torch.empty((num_samples)).to(num_samples.device))
        z_samples = self.sample_latent(mu_prior, sigma_prior, K=1)
        return self.generative_distr(z_samples)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([torch.prod(torch.tensor(p.size())).item() for p in model_parameters])
        return super().__str__() + f"\nTrainable parameters: {params:_}"


class MLP_VAE(BaseVAE):
    def __init__(self, d, D, sigma_loss, encoder_hidden, encoder_num, decoder_hidden, decoder_num, clamp):
        super().__init__(d, D, sigma_loss)
        lin_D = D[0] * D[1] * D[2]
        
        self.proposal_network = nn.Sequential(
            nn.Linear(lin_D, encoder_hidden),
            nn.BatchNorm1d(encoder_hidden),
            nn.LeakyReLU(),
            *chain(
                *[[
                    nn.Linear(encoder_hidden, encoder_hidden), 
                    nn.BatchNorm1d(encoder_hidden),
                    nn.LeakyReLU()
                ] for i in range(encoder_num - 1)]
            )
        )
        self.proposal_mu_head = nn.Linear(encoder_hidden, self.d)
        self.proposal_sigma_head = nn.Sequential(
            nn.Linear(encoder_hidden, self.d),
            nn.Softplus()
        )
        self.generative_network = nn.Sequential(
            nn.Linear(self.d, decoder_hidden),
            nn.BatchNorm1d(decoder_hidden),
            nn.LeakyReLU(),
            *chain(
                *[[
                    nn.Linear(decoder_hidden, decoder_hidden), 
                    nn.BatchNorm1d(decoder_hidden),
                    nn.LeakyReLU(),
                ] for i in range(decoder_num - 2)]
            ),
            nn.Linear(decoder_hidden, lin_D), 
            ClampLayer(-clamp, clamp),
            # nn.Sigmoid()
        )

    def proposal_distr(self, x):
        """
        Generate proposal distribution over z.
        Note that sigma is positive by design of neural network.
        Input: x, Tensor of shape n x D.
        Return: tuple(Tensor, Tensor),
                Each Tensor is a matrix of shape n x d.
                The first one is mu, the second one is sigma.
        """
        x = x.reshape(x.shape[0], -1)
        hidden = self.proposal_network(x)
        mu = self.proposal_mu_head(hidden)
        sigma = self.proposal_sigma_head(hidden)
        return mu, sigma
    
    def generative_distr(self, z):
        """
        Compute a tensor of parameters of Bernoulli distribution over x
        given a tensor of latent representations.
        Input: z, Tensor of shape n x K x d - tensor of latent representations.
        Return: Tensor of shape n x K x D - parameters of Bernoulli distribution.
        """

        params = self.generative_network(z.reshape(z.shape[0] * z.shape[1], -1)) # n*k, prod(D)
        params = params.reshape(z.shape[0], z.shape[1], *self.D) # n, k, d1 ... dn
        return params
