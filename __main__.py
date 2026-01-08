from src.rxn_fit import from_json_input
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', 
    '--input', 
    type=str, 
    required=True, 
    help='''
        input json directory for fitting configuration
        e.g.
        {
            "data": "data.csv",
            "reactions": [
                "A+B=C+D",
                "C+D=E"
            ],
            "resolution": 50,
            "method": "euler_i",
            "epoch": 1000,
            "units": {
                "time": "min",
                "molarity": "mM"
            }
        }
    '''
)
parser.add_argument(
    '-v', 
    '--verbose', 
    action='store_true',
    help='report procedure'
)
args = parser.parse_args()
from_json_input(args.input, args.verbose)