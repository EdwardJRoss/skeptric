#!/usr/bin/env python
import argparse
from collections import Counter
from pathlib import Path
import yaml

EXTENSIONS = ['Rmd', 'qmd', 'md']
BLOCK_SEPARATOR = '---'

def get_frontmatter(lines):
    result = ''
    in_block = False
    for line in lines:
        if line.startswith(BLOCK_SEPARATOR):
            if in_block:
                break
            else:
                in_block = True
                continue
        if in_block:
            result += line
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("category")
    args = parser.parse_args()

    category = args.category

    for path in [path for ext in EXTENSIONS for path in Path(".").glob(f"**/*.{ext}")]:
        with open(path) as f:
            frontmatter_text = get_frontmatter(f)
            frontmatter_data = yaml.safe_load(frontmatter_text)
            if category in frontmatter_data.get('categories', []):
                print(path)



