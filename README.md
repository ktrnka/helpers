I found that I was copying and pasting code between machine learning projects so this is an attempt to
simplify by having a shared Git submodule that I can update.

Goals
=====

* Should be able to use most of the package even if Theano isn't installed (keep those dependencies separate cause I couldn't get Theano working on Windows)
* Shared library = enough incentive to write unit tests for common functionality
* Sharing some of my helper code back to the community

Issues
======

* I can't seem to get relative imports to work for both the unit tests and for the project I'm using this in.
This is a major weakness.
