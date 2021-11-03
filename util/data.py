import torch

##########################################################################################
#                                   The training data
##########################################################################################


# TODO: simplify with a dict with all the available distributions, e.g.:

distributions_dict = {
    'bernoulli': torch.distributions.bernoulli.Bernoulli,
    'beta': torch.distributions.beta.Beta,
    'normal': torch.distributions.normal.Normal,
    # etc
}

# TODO: then we sample from it with a sample function (w/device, loc, var, anything that it needs, with default values)
# Should also save the data to a .npy file to be referenced afterwards if needed/for reproducibility purposes

class NormalDistribution:
    """Generate data with a Normal distribution; N(mu=loc, var=scale^2)"""
    def __init__(self, loc=4.0, scale=0.5, device='cpu'):
        self.device = device if ('cuda' in device and torch.cuda.is_available()) else 'cpu'
        self.loc = torch.tensor([loc], device=self.device)
        self.scale = torch.tensor([scale], device=self.device)

    def sample(self, num_samples, seed=42):
        # Set the seed for reproducibility reasons
        torch.manual_seed(seed)
        # Define the distribution
        m = torch.distributions.normal.Normal(loc=self.loc, scale=self.scale)
        # Sample num_samples numbers from this distribution
        samples = m.sample([num_samples])
        return samples


class UniformDistribution:
    """Generate data with a Uniform distribution; U([low, high])"""
    def __init__(self, low=-2.3, high=3.3, device='cpu'):
        self.device = device if ('cuda' in device and torch.cuda.is_available()) else 'cpu'
        self.low = torch.tensor([low], device=self.device)
        self.high = torch.tensor([high], device=self.device)

    def sample(self, num_samples, seed=42):
        # Set the seed for reproducibility reasons
        torch.manual_seed(seed)
        # Define the distribution
        m = torch.distributions.Uniform(low=self.low, high=self.high)
        # Sample num_samples numbers from this distribution
        samples = m.sample([num_samples])
        return samples


class LaplaceDistribution:
    """Generate data with a Laplace distribution: Laplace(mu=loc, b=scale)"""
    def __init__(self, loc=3.0, scale=0.3, device='cpu'):
        self.device = device if ('cuda' in device and torch.cuda.is_available()) else 'cpu'
        self.loc = torch.tensor([loc], device=self.device)
        self.scale = torch.tensor([scale], device=self.device)

    def sample(self, num_samples, seed=42):
        # Set the seed for reproducibility reasons
        torch.manual_seed(seed)
        # Define the distribution
        m = torch.distributions.Laplace(loc=self.loc, scale=self.scale)
        # Sample num_samples numbers from this distribution
        samples = m.sample([num_samples])
        return samples


class HalfNormalDistribution:
    """Generate data with a Half Normal distribution: N(mu=0, var=scale^2)"""
    def __init__(self, scale=0.75, device='cpu'):
        self.device = device if ('cuda' in device and torch.cuda.is_available()) else 'cpu'
        self.scale = torch.tensor([scale], device=self.device)

    def sample(self, num_samples, seed=42):
        # Set the seed for reproducibility reasons
        torch.manual_seed(seed)
        # Define the distribution
        m = torch.distributions.HalfNormal(scale=self.scale)
        # Sample num_samples numbers from this distribution
        samples = m.sample([num_samples])
        return samples


class PetitPrinceDistribution:
    """Generate data with a 'Petit Prince' distribution: N(mu=4.0, var=1.5^2) + N(mu=4.0, var=1.35^2)"""
    def __init__(self, loc1=4.0, scale1=1.5, loc2=0.6, scale2=1.35, device='cpu'):
        self.device = device if ('cuda' in device and torch.cuda.is_available()) else 'cpu'
        self.loc1 = torch.tensor([loc1], device=self.device)
        self.scale1 = torch.tensor([scale1], device=self.device)
        self.loc2 = torch.tensor([loc2], device=self.device)
        self.scale2 = torch.tensor([scale2], device=self.device)

    def sample(self, num_samples, seed=42):
        # Set the seed for reproducibility:
        torch.manual_seed(seed)
        # Define the distributions:
        m1 = torch.distributions.normal.Normal(loc=self.loc1, scale=self.scale1)
        m2 = torch.distributions.normal.Normal(loc=self.loc2, scale=self.scale2)
        # Sample num_samples numbers from this distribution
        samples = torch.cat((m1.sample([num_samples//2]), m2.sample([num_samples - num_samples//2])), 0)
        return samples
