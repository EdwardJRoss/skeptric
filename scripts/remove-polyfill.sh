#!/usr/bin/env bash
set -euo pipefail

while IFS= read -r output_file; do
  case "$output_file" in
    *.html)
      perl -0pi -e 's/\n\s*<script src="https:\/\/polyfill\.io\/v3\/polyfill\.min\.js\?features=es6"><\/script>//g' "$output_file"
      ;;
  esac
done <<< "${QUARTO_PROJECT_OUTPUT_FILES:-}"
