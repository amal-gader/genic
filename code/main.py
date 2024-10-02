import argparse

from gen_ic import train_genic
from evaluate_gen_ic import test_genic

def train(args):
    print(f"Training on dataset: {args.dataset}")
    print(f"With Type: {args.with_type}")
    print(f"With Description: {args.with_desc}")
    train_genic(args)
    
    
def test(args):
    print(f"Testing with dataset: {args.dataset}")
    print(f"Link Prediction Checkpoint: {args.id_lp}")
    print(f"Property Prediction Checkpoint: {args.id_pp}")
    print(f"With Type: {args.with_type}")
    print(f"With Description: {args.with_desc}")
    test_genic(args)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help="Choose the mode: 'train' or 'test'")

    # arguments for both train and test
    parser.add_argument('--dataset', type=str, default=None, choices=['codex', 'fb15k', 'wn18rr'], help='The dataset to be used: codex, fb15k, or wn18rr.')
    parser.add_argument('--with_type', action='store_true', help="Include entity types")
    parser.add_argument('--with_desc', action='store_true', help="Include entity descriptions")

    # specific to the test mode
    parser.add_argument('--id_lp', type=str, help='Checkpoint id to use for link prediction (test mode only)')
    parser.add_argument('--id_pp', type=str, help='Checkpoint id to use for property prediction (test mode only)')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
