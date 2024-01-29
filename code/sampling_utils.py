from tqdm import tqdm

def nrange(n_its, bar):
    """Auxiliary function, not to be called by the user"""
    if bar:
        return tqdm(range(1, n_its+1))
    return range(1, n_its+1)

