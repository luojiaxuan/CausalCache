#!/usr/bin/env bash
set -euo pipefail

# note (luojiaxuan): Build a minimal arXiv source archive whose root main.tex selects the named-author preprint path.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
paper_dir="$repo_root/paper"
output_dir="$repo_root/output/arxiv"
staging_root="$(mktemp -d "${TMPDIR:-/tmp}/causalcache-arxiv.XXXXXX")"
trap 'rm -rf "$staging_root"' EXIT

mkdir -p "$output_dir" "$staging_root/source/figures"
awk 'BEGIN { print "\\def\\ARXIVVERSION{1}" } { print }' \
  "$paper_dir/main.tex" > "$staging_root/source/main.tex"
cp "$paper_dir/references.bib" "$staging_root/source/references.bib"
cp "$paper_dir/aaai2027.sty" "$staging_root/source/aaai2027.sty"
cp "$paper_dir/aaai2027.bst" "$staging_root/source/aaai2027.bst"
cp "$paper_dir/figures/causalcache_overview_case.pdf" "$staging_root/source/figures/causalcache_overview_case.pdf"
cp "$paper_dir/figures/causalcache_hgkv_pass2.pdf" "$staging_root/source/figures/causalcache_hgkv_pass2.pdf"
cp "$paper_dir/figures/causalcache_split_bars.pdf" "$staging_root/source/figures/causalcache_split_bars.pdf"
cp "$paper_dir/figures/causalcache_budget_curve.pdf" "$staging_root/source/figures/causalcache_budget_curve.pdf"

COPYFILE_DISABLE=1 tar -C "$staging_root/source" -czf "$output_dir/causalcache_arxiv_source.tar.gz" .
