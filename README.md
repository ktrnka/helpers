I found that I was copying and pasting code between machine learning projects so this is an attempt to
simplify by having a shared Git submodule that I can update.

Goals
=====

* Should be able to use most of the package even if Theano isn't installed (keep those dependencies separate cause I couldn't get Theano working on Windows)
* Shared library = enough incentive to write unit tests for common functionality
* Sharing some of my helper code back to the community

TODO
======

* requirements.txt
* Merge some of the ensemble work from my League project
* Test and utilize the debug module
* Can I merge pairwise feature generation? In the other code base it's dependent on a base model.