import torch

##########################################################################################
################################### The training data ####################################
##########################################################################################


class NormalDistribution:
    def __init__(self, loc=4.0, scale=0.5):
        self.loc = torch.tensor([loc])
        self.scale = torch.tensor([scale])

    def sample(self, num_samples, seed=42):
        # Set the seed for reproducibility reasons
        torch.manual_seed(seed)
        # Define the distribution
        m = torch.distributions.normal.Normal(loc=self.loc, scale=self.scale)
        # Sample n numbers from this distribution
        samples = m.sample([num_samples])
        return samples


class LaplaceDistribution:
    def __init__(self, loc=3.0, scale=0.3):
        self.loc = torch.tensor([loc])
        self.scale = torch.tensor([scale])

    def sample(self, num_samples, seed=42):
        # Set the seed for reproducibility reasons
        torch.manual_seed(seed)
        # Define the distribution
        m = torch.distributions.Laplace(loc=self.loc, scale=self.scale)
        # Sample n numbers from this distribution
        samples = m.sample([num_samples])
        return samples


class HalfNormalDistribution:
    def __init__(self, scale=0.75):
        self.scale = torch.tensor([scale])

    def sample(self, num_samples, seed=42):
        # Set the seed for reproducibility reasons
        torch.manual_seed(seed)
        # Define the distribution
        m = torch.distributions.HalfNormal(scale=self.scale)
        # Sample n numbers from this distribution
        samples = m.sample([num_samples])
        return samples


class PetitPrinceDistribution:
    def __init__(self, loc1=4.0, scale1=1.5, loc2=0.6, scale2=1.35):
        self.loc1 = torch.tensor([loc1])
        self.scale1 = torch.tensor([scale1])
        self.loc2 = torch.tensor([loc2])
        self.scale2 = torch.tensor([scale2])

    def sample(self, num_samples, seed=42):
        # Set the seed for reproducibility:
        torch.manual_seed(seed)
        # Define the distributions:
        m1 = torch.distributions.normal.Normal(loc=self.loc1, scale=self.scale1)
        m2 = torch.distributions.normal.Normal(loc=self.loc2, scale=self.scale2)
        # Sample n numbers from this distribution
        samples = torch.cat((m1.sample([num_samples//2]), m2.sample([num_samples-num_samples//2])), 0)
        return samples
