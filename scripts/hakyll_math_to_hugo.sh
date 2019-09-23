#!/bin/bash
set -exuo pipefail

POSTS='../content/post'


for file in "${POSTS}"/*.md; do
    sed -i 's/\$/$$/g' "${file}"
    mv "$file" "${file/.md/.mmark}"
done
