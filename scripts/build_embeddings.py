"""Encode every prepared pair with one pinned backbone; keep caches private."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.encoder_variants import cache_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', default='data/processed/semrel-full-v3')
    parser.add_argument('--variant', choices=('afro', 'e5'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cache_dataset(args.processed, args.output, args.variant, max_tokens=512)
