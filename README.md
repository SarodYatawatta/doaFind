# doaFind
Direction of arrival estimation using radio interferometric arrays. Methods are described in [this paper](https://arxiv.org/abs/2510.15116).

Run from *./src* directory. Pass *A12* or *SKA* for *--array* option.

Generate training data:

```
./rfisig.py --seed 1 --array A12
```

Train model:

```
./train.py --iterations 250000
```

Generate testing data:

```
./rfisig.py --seed 2 --array A12
```

Evaluate model:

```
./eval.py --iterations 20
```

Pass *--help* option to see more options.


do 11 sep 2025  0:52:12 CEST
