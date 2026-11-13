# R3 revision package with comment tracking

Every change in the manuscript is tagged with the reviewer comment that it answers, and every value is final.

## Comment identifiers and colours

- R5C1 to R5C12 are the twelve comments of Reviewer 5, and R7C1 to R7C7 are the comments of Reviewer 7.
- Blue text answers Reviewer 5, and red text answers Reviewer 7.
- Purple text answers a concern that both reviewers raised (R5C1 with R7C5, R5C8 with R7C1, and R5C9 with R7C2), so it carries both identifiers.
- Green text is copy-editing, recorded under R7C7.
- Deleted text is struck through in the same colour.
- The last page of the tracked manuscript has an index that lists, for every comment, the pages on which it is addressed. The response letter opens with the same matrix and gives the page list under each comment. These page numbers are generated automatically from the tags.

## manuscript/ (upload these files)

- `manuscript_R3_tracked.tex`: the tracked version.
- `manuscript_R3_clean.tex`: the clean version. It is built from the same source with `\trackedfalse`, so its text is identical to the tracked version without the mark-up.
- `response_to_reviewers_R3.tex`: the point-by-point response, with the R5Cx and R7Cx identifiers.
- `references.bib` and `fig1_framework.png` to `fig8_fa_latency.png`.
- `manuscript_R3_tracked.aux`: the page record of the tracked manuscript, used by the response letter.

The tracking macros are embedded in the preamble of both manuscript files, so no extra style file is needed. Strike-through uses `ulem`, which Overleaf provides.

## Compiling

The files compile with the Elsevier CAS class (`cas-sc.cls`, `cas-model2-names.bst`), which Overleaf already provides. They were tested with the real class, with all files in the project root and also with the files kept inside a `manuscript/` subfolder, so you do not need to change the figure or bibliography paths. Do not add a second `\usepackage{natbib}` line, because the preamble already loads natbib with the author-year options and a second load causes an option clash.

On your own computer, compile in this order:

```
pdflatex manuscript_R3_tracked
bibtex   manuscript_R3_tracked
pdflatex manuscript_R3_tracked
pdflatex manuscript_R3_tracked
pdflatex response_to_reviewers_R3
pdflatex response_to_reviewers_R3
```

Compile `manuscript_R3_clean.tex` the same way: pdflatex, bibtex, pdflatex, pdflatex.

On Overleaf, the response letter reads its page numbers from `manuscript_R3_tracked.aux`. A copy of that file is included, so the letter compiles with correct page numbers straight away. If the pages move after you paste your author block, compile the tracked manuscript, download `output.aux` from "Logs and output files", rename it to `manuscript_R3_tracked.aux`, replace the included copy, and compile the letter again.

Before submitting, paste your author and affiliation block into both manuscript files at the line that says so.

## supplementary/ and code/

These are unchanged from the previous package. They contain the audit, the Protocol B re-run with all 30 runs, the Holm tests, the routing validation, the baselines, the Colab notebook and the scripts.

## preview/

These PDFs were compiled with the real Elsevier CAS class, without your author block. Submit the PDFs that you compile yourself after adding the author block.
