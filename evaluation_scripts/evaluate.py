"""Explicit final evaluation, separate from training and inference."""
from pathlib import Path
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

if __name__=='__main__':
    module = 'src.models.full_evaluation' if len(sys.argv)>1 and sys.argv[1] in ('freeze','release') else 'src.models.evaluate'
    runpy.run_module(module,run_name='__main__')
