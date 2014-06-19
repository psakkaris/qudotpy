QuDotPy
=======

A quantum computing library written in Python. Exploring quantum computing has never been easier. With QuDotPy you can 
experiment with single-qubit operations and gates. You can build multiple-qubit states and perform measurements, and finally you can emulate quantum circuits. Our quantum circuit design makes it easy to test out circuit ideas and to step through a circuit like you would a debugger.

To help you get started we have written a detailed usage tutorial that covers most aspects of QuDotPy. The tutorial can be found here: http://psakkaris.github.io/qudotpy/

QuDotPy depends on Numpy. You will need to have Numpy installed before you can use QuDotPy. 


Getting Started
===============

You can always clone this repository and check out the code. However, if you just want to get a feel for QuDotPy just press the 'Download Zip' button on the right panel. This will download the directory **qudotpy-master**

Then you should make a link on your system (for *nix and OS X):
```
ln -s /Users/softwaretest/code/qudotpy-master/ 
      /Users/softwaretest/code/qudotpy
```

Next run the pyton shell from the directory qudotpy is located and give it a try

```
$ python
>>> from qudotpy import qudot
>>> print qudot.apply_gate(qudot.H, qudot.ZERO)
(0.707106781187+0j)|0> + (0.707106781187+0j)|1>

>>> 

```

That's it! For more check out our tutorial http://psakkaris.github.io/qudotpy/

Giving Back
===========

Feel free to contribute back to QuDotPy. If you are new to GitHub read this: 

https://help.github.com/articles/fork-a-repo

If you find a bug, would like to suggest an enhancement or would like to see enhancement we have planned for the future; check out the Issues page.
