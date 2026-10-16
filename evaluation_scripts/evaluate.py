"""Explicit final evaluation, separate from training and inference."""
from pathlib import Path
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

if __name__=='__main__':runpy.run_module('src.models.evaluate',run_name='__main__')
