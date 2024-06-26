# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(2, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../../'))
# sys.path.insert(0, os.path.abspath('../../pymultipact'))
# sys.path.insert(0, os.path.abspath('../../pymultipact/matlab'))


# -- Project information -----------------------------------------------------

project = 'PyMultipact'
copyright = '2024, Sosoho-Abasi Udongwo'
author = 'Sosoho-Abasi Udongwo'

# The full version, including alpha/beta/rc tags
release = '05.05.2024'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'numpydoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.inheritance_diagram',
    'matplotlib.sphinxext.mathmpl',
    'sphinx.ext.autosectionlabel',
    'sphinxcontrib.matlab'
    # 'sphinxcontrib.bibtex'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
bibtex_bibfiles = ['../references/refs.bib']
# primary_domain = "mat"
print(os.path.dirname(os.path.abspath(r'D:\Dropbox\PyMultipact\pymultipact\matlab')))
matlab_src_dir = os.path.dirname(os.path.abspath(r'D:\Dropbox\PyMultipact\pymultipact\matlab'))

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinxdoc'
# html_theme = 'bootstrap'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "./images/logo.png"
# html_static_path = sphinx_bootstrap_theme.get_html_theme_path()

# Enable numref
numfig = True

numpydoc_show_class_members = False