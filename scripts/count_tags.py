#!/usr/bin/env python
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
    paths = [path for ext in EXTENSIONS for path in Path(".").glob(f"**/*.{ext}")]

    meta = []

    for path in paths:
        with open(path) as f:
            frontmatter_text = get_frontmatter(f)
            frontmatter_data = yaml.safe_load(frontmatter_text)
            meta.append(frontmatter_data)

    category_counts = Counter(cat for m in meta for cat in m.get('categories', []) if not m.get('draft'))

    for tag, count in category_counts.most_common():
        print(f"{tag: <16} {count}")



