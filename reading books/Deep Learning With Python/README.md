> notes and codes from reading [this book](https://www.manning.com/books/deep-learning-with-python)

# Notes
## ch1 What is deep learning?
* derive useful input representation that gets us closer to the expected outupt
* automatically find such transformation that turns data into more useful representation for a given task
* ML = searching for userful representation of some input data, within a predefined space of possibilities, using guidance from some feedback signal
* a network with minumum loss is one for which the outputs are as close as it can be to the target

## ch2 Math
* (2.3.6) Neural networks consist entirely in chains of tensors operations, and that all these tensor operations are really just geometric transformations of the input data. It follows that you can interpret a neural network as a very complex geometric transformation in a high-dimensional space, implemented via a long series of simple steps.
* Imagine two sheets of colored paper, a red one and a blue one. Superpose them. Now crumple them together into a small paper ball. That crumpled paper ball is your input data, and each sheet of paper is a class of data in a classification problem. What a neural network (or any other machine learning model) is meant to do, is to figure out a transformation of the paper ball that would uncrumple it, so as to make the two classes cleanly separable again.