# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    ?= -j auto -WT --keep-going
SPHINXBUILD   ?= sphinx-build
SPHINXPROJ    ?= TrojanZoo
SOURCEDIR     ?= source
BUILDDIR      ?= build
PYCMD         ?= python

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# docset: html
# 	doc2dash --name $(SPHINXPROJ) --icon $(SOURCEDIR)/_static/img/pytorch-logo-flame.png --enable-js --online-redirect-url https://pytorch.org/docs/ --force $(BUILDDIR)/html/

.PHONY: help Makefile docset

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile figures
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@echo "Removing everything under 'build' and 'source/generated'.."
	@rm -rf $(BUILDDIR)/html/ $(BUILDDIR)/doctrees $(SOURCEDIR)/generated