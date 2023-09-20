from pipeline import scanner, scanner_and_visualize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_path')
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

if args.visualize:
    scanner_and_visualize(args.image_path)
else:
    print(scanner(args.image_path))

