"""Train complete-cohort encoder heads; test targets remain closed."""
import argparse

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-afro');parser.add_argument('--cache-e5')
    parser.add_argument('--cache',help='Optional single-cache compatibility command')
    parser.add_argument('--prior',help='Fresh matched-encoder study for the fixed score-prior comparison')
    parser.add_argument('--processed',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--protocol')
    args=parser.parse_args()
    if args.cache:
        if args.cache_afro or args.cache_e5 or args.prior:parser.error('Use either a single cache or both matched caches')
        from src.models.train import train
        train(args.cache,args.processed,args.output)
    else:
        if not args.cache_afro or not args.cache_e5:parser.error('Both --cache-afro and --cache-e5 are required')
        if args.prior:
            if not args.protocol:parser.error('--prior requires an explicitly frozen fresh --protocol')
            from src.models.score_prior import train
            train(args.cache_afro,args.cache_e5,args.processed,args.prior,args.protocol,args.output)
        else:
            from src.models.sentence_comparison import train
            train(args.cache_afro,args.cache_e5,args.processed,args.output,args.protocol or 'docs/full-cohort-v3-protocol.json')
