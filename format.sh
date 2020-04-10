black -l 79 bugs/
black -l 79 examples/
black -l 79 tensorfuzz/
pylint bugs/*.py
pylint tensorfuzz/*.py
pylint examples/dcgan/*.py
pylint examples/nans/*.py
pylint examples/quantize/*.py
