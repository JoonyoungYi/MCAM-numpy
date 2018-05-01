# Low-rank Matrix Completion using Alternating Minimization(MCAM-numpy)

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
virtualenv .venv -p python3
. .venv/bin/activate
pip install -r requirements.txt
python run.py
```

- The example of result(console)
```
RANK of M        : 2
|U_hat-U|_F/|U|_F: 1.4007495010575732

>> t(  0): 7.056300795604675e-17
>> t(  1): 8.023918248411439e-17
>> t(  2): 6.838685255066973e-17
>> t(  3): 7.616046824035152e-17
>> t(  4): 6.07257413689918e-17

RANK of X        : 2
TRAIN RMSE       : 1.9046232036157855e-17
TEST  RMSE       : 6.635452802602127e-18
|X-M|_F/|M|_F    : 4.614953476853536e-16
```
