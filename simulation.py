import numpy as np

def bayesian_integration(fingers, prior, likelihood, n_neurons, noise,):
    # 2. Define prior belief over fingers
    prior = np.array([prior, 1 - prior])  # uniform prior; could be biased if desired

    # 3. Define likelihood for some sensory input x
    # Let's say the sensory evidence x favors index
    # e.g., the sensory system detects a deflection consistent with index

    # Likelihood of observing x given each finger
    likelihood = np.array([1 - noise if likelihood == 1 else noise,
                           1 - noise if likelihood == 0 else noise])  # P(x|index), P(x|ring)

    # 4. Compute posterior via Bayes' rule
    posterior = prior * likelihood
    posterior /= np.sum(posterior)

    print(dict(zip(fingers, posterior, )))

    # 5. Create population
    preferred_finger = np.array(['index'] * (n_neurons // 2) + ['ring'] * (n_neurons // 2))

    # Map posterior probabilities to firing rates
    firing_rates = np.array([
        posterior[0] if pf == 'index' else posterior[1]
        for pf in preferred_finger
    ])

    # Add Poisson-like noise
    firing_rates += np.random.poisson(lam=5, size=firing_rates.shape) / n_neurons

    return firing_rates


def signed_error(fingers, prior, stimulated_finger, n_neurons):
    """
    Neurons encode signed error between prior belief and actual stimulated finger.
    Index neurons increase firing for positive error; ring neurons for negative error.
    """

    # Compute signed error: prior index prob - actual stimulation (0 or 1)
    error = prior - stimulated_finger  # could be negative or positive

    # Create population
    preferred_finger = np.array(['index'] * (n_neurons // 2) + ['ring'] * (n_neurons // 2))

    # Map error to firing rates: index neurons respond to +error, ring to -error
    firing_rates = np.array([
        error if pf == 'index' else error
        for pf in preferred_finger
    ], dtype=float)

    # Add noise
    firing_rates += np.random.poisson(lam=5, size=firing_rates.shape) / n_neurons

    return firing_rates


def unsigned_error(fingers, prior, stimulated_finger, n_neurons):
    """
    Neurons encode signed error between prior belief and actual stimulated finger.
    Index neurons increase firing for positive error; ring neurons for negative error.
    """

    # Compute signed error: prior index prob - actual stimulation (0 or 1)
    error = np.abs(prior - stimulated_finger)  # could be negative or positive

    # Create population
    preferred_finger = np.array(['index'] * (n_neurons // 2) + ['ring'] * (n_neurons // 2))

    # Map error to firing rates: index neurons respond to +error, ring to -error
    firing_rates = np.array([
        error if pf == 'index' else error
        for pf in preferred_finger
    ], dtype=float)

    # Add noise
    firing_rates += np.random.poisson(lam=5, size=firing_rates.shape) / n_neurons

    return firing_rates