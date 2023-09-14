from pipeline import scanner
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_path')
args = parser.parse_args()

print(scanner(args.image_path))

