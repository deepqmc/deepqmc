BLDDIR = _build
export max_print_line = 1000000

default: preprint

all: preprint si figs

preprint: $(BLDDIR)/preprint.pdf

si: $(BLDDIR)/si.pdf

figs:
	@mkdir -p $(BLDDIR)
	pdflatex -output-directory=$(BLDDIR) -shell-escape export-figures.tex

diff: $(BLDDIR)/diff.pdf

bib: refs-zotero.bib

$(BLDDIR)/%.pdf: %.tex refs-zotero.bib FORCE
	latexmk -pdf -outdir=$(BLDDIR) -interaction=nonstopmode $<

$(BLDDIR)/preprint.pdf: preamble.tex
$(BLDDIR)/si.pdf: preamble.tex

refs-zotero.bib: FORCE
	-curl -sfo $@ "http://127.0.0.1:23119/better-bibtex/collection?/1/ML/DL-QMC.bibtex&exportNotes=false&useJournalAbbreviation=true" && gsed -i '/./,$$!d' $@

clean:
	rm -rf $(BLDDIR)

distclean: clean
	rm -f $(patsubst %.tex.in,%.tex,$(wildcard *.tex.in))

FORCE:
