# Low-rank Matrix Completion using Alternating Minimization(lrmc-am)

- The repository reproducing algorithm in the paper, [Low-rank Matrix Completion using Alternating Minimization](https://arxiv.org/abs/1212.0467).
  - I reproduce the Algorithm 2(AltMinComplete) only. Because, Algorithm 1 is not for matrix completion problem, but for matrix sensing problem.

- In this repository, I used numpy to implement the paper.

- The meanings of parameters.
```
m: The number of row of the matrix
n: The number of column of the matrix
p: samling probability
omega: the observed set omega. if entries are given, 1 else 0.
cardinality_of_omega: The cardinality of set omega.

M: The matrix to recover.
X: The solution of AM.

k: The rank of the matrix.
T: The number of iteration
mu: coherence
```

- How to init and run
```
cd np
virtualenv .venv -p python3
. .venv/bin/activate
pip install -r requirements.txt
python run.py
```
