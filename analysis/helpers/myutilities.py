from itertools import islice

def dict_head(d, n=5):
    """Prints the first n key-value pairs in a dictionary."""
    return dict(islice(d.items(), n))