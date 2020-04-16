# code for question on reddit https://www.reddit.com/r/MachineLearning/comments/8poc3z/r_blog_post_on_world_models_for_sonic/e0cwb5v/
# from this

def forward(self, x):
    self.lstm.flatten_parameters()

    x = F.relu(self.fc1(x))
    z, self.hidden = self.lstm(x, self.hidden)
    sequence = x.size()[1]

    # predict log pi instead and use log softmax
    pi = self.z_pi(z).view(-1, sequence, self.n_gaussians, self.z_dim)
    pi = F.softmax(pi, dim=2)
    sigma = torch.exp(self.z_sigma(z)).view(-1, sequence,
                    self.n_gaussians, self.z_dim)
    mu = self.z_mu(z).view(-1, sequence, self.n_gaussians, self.z_dim)
    return pi, sigma, mu

def mdn_loss_function(out_pi, out_sigma, out_mu, y):
    y = y.view(-1, SEQUENCE, 1, LATENT_VEC)
    result = Normal(loc=out_mu, scale=out_sigma)

    # replace with logsumexp here, we have to mvoe the result*out_pi earlier
    result = torch.exp(result.log_prob(y))
    result = torch.sum(result * out_pi, dim=2)   # do this earlier using log rule `log(x*y) = log(x) + log(y)`
    result = -torch.log(EPSILON + result)
    return torch.mean(result)


# Into this?



def logsumexp(x, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    from https://github.com/pytorch/pytorch/issues/2591

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = torch.where(
        (xm == float('inf')) | (xm == float('-inf')), 
        xm,
        xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    return x if keepdim else x.squeeze(dim)


def forward(self, x):
    self.lstm.flatten_parameters()

    x = F.relu(self.fc1(x))
    z, self.hidden = self.lstm(x, self.hidden)
    sequence = x.size()[1]

    # predict log pi instead, as it's often more stable to predict logprob than prob
    log_pi = self.z_pi(z).view(-1, sequence, self.n_gaussians, self.z_dim)
    log_pi = F.log_softmax(log_pi, dim=2)
    sigma = torch.exp(self.z_sigma(z)).view(-1, sequence,
                    self.n_gaussians, self.z_dim)
    mu = self.z_mu(z).view(-1, sequence, self.n_gaussians, self.z_dim)
    return log_pi, sigma, mu

def mdn_loss_function(log_pi, out_sigma, out_mu, y):
    y = y.view(-1, SEQUENCE, 1, LATENT_VEC)
    result = Normal(loc=out_mu, scale=out_sigma)
    log_prob = result.log_prob(y)

    # mutliply rho/pi/weight earlier using the log rule `log(x*y) = log(x) + log(y)`
    # so that we can use the more stable logsumexp on the result
    weighted_log_prob = log_prob + log_pi
    result = - logsumexp(weighted_log_prob, dim=2)
    return torch.mean(result)

