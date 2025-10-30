# doaFind
Direction of arrival estimation using radio interferometric arrays. Methods are described in [this paper](https://arxiv.org/abs/2510.15116).

Run from *./src* directory. Pass *A12* or *SKA* for *--array* option.

1. Generate training data:

```
./rfisig.py --seed 1 --array A12
```

2. Train model:

```
./train.py --iterations 250000
```

3. Generate testing data:

```
./rfisig.py --seed 2 --array A12
```

4. Evaluate model:

```
./eval.py --iterations 20
```

Pass *--help* option to see more options.

Use *--simulate_range*, *--estimate_range* and *--range_grid* options to perform DOA and range estimation.

do 30 okt 2025  9:34:37 CET
