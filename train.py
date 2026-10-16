"""Train frozen-encoder symmetric regressors; test gold is never opened."""
from src.models.train import train

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','processed','output'):parser.add_argument('--'+name,required=True)
    train(**vars(parser.parse_args()))
