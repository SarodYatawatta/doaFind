# doaFind
Direction of arrival estimation using radio interferometric arrays


Run from *./src* directory. Pass *A12* or *SKA* for *--array* option.

Generate training data:

```
./rfisig.py --seed 1 --array A12
```

Train model:

```
./train.py --iterations 250000
```

Evaluate model:

```
./eval.py --iterations 20
```

Pass *--help* option to see more options.


do 11 sep 2025  0:52:12 CEST
