# Configuration file for the Sphinx documentation builder.


# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# adjusting the path to the root directory
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'dynml'
author = 'Gregory R. Macchio'
copyright = '2025, Gregory R. Macchio'


# add autodoc and rtd theme extensions
extensions = ['sphinx.ext.autodoc', 'sphinx_rtd_theme']


# make sure autodoc generates documentation in the order it is coded
autodoc_member_order = 'bysource'


# make sure autodoc skips properties
def skip_properties(app, what, name, obj, skip, options):
    # Skip properties
    if isinstance(obj, property):
        return True
    return skip
def setup(app):
    app.connect('autodoc-skip-member', skip_properties)


# template path and excluded patterns
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# options for html output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_rtd_theme"
html_theme_options = {'navigation_depth': 10}


# mocking the import of torch because it is not installed on readthedocs
autodoc_mock_imports = ["torch", "tqdm"]
