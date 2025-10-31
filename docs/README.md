# PyTorch Hierarchical Loss

This package provides functions to compute Binary Cross Entropy loss for hierarchical categories.

The general strategy is to use the "flat" predictions from a general detection or classification model.
  These predictions are then interpreted as *conditional* logit confidences.
  This allows us to use the existing model architecture as-is, and interpret the output as hierarchical only changing how we interpret the predictions.

More specifically, suppose that we have $n$ categories, and a hierarchical structure over these categories.  Suppose that our model predicts a vector $V$ for some object or image.  We interpret the logit value in index i to be:

$$ V[i] := logit(P(\text{category}[i] | \text{parent}(\text{category}[i]))) $$

We can compute the raw conditional probability using a sigmoid function.

$$ sigmoid(V[i]) := P(\text{category}[i] | \text{parent}(\text{category}[i])) $$

We can use these to compute marginal confidences at arbitrary locations in the hierarchy:

$$ P(\text{category}[i]) = P(\text{category}[i] | \text{parent}(\text{category}[i])) * P(\text{parent}(\text{category}[i]) | \text{parent}(\text{parent}(\text{category}[i]))) \ldots $$

Given these marginal confidences, we can compute the ordinary BCE loss.

Likely, the point of entry users will be most interested in is the `Hierarchy` class, which handles caching various views of the hierarchy, and the `hierarchical_loss` function.  You will need to incorporate the loss into your existing training regimen.  After training, one may find useful the `optimal_hierarchical_path` and various `...truncate...` functions in `path_utils` for predictive purposes.  Note that the standard predictive "choose the category with the highest confidence" strategy does not work here, since the marginal probability of a child will always necessarily be less than the marginal probability of the parent.  

# Installation

    pip install git+https://github.com/csbrown-noaa/hierarchical_loss.git

# Contributing

We would love to have your contributions that improve current functionality, fix bugs, or add new features.  See [the contributing guidelines](CONTRIBUTING.md) for more info.

# Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and
Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project
code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the
Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub
project will be governed by all applicable Federal law. Any reference to specific commercial products,
processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or
imply their endorsement, recommendation or favoring by the Department of Commerce. The Department
of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to
imply endorsement of any commercial product or activity by DOC or the United States Government.
