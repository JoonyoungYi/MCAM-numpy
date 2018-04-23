# Low-rank Matrix Completion using Alternating Minimization(lrmc-am)

- The repository reproducing algorithm in the paper, [Low-rank Matrix Completion using Alternating Minimization](https://arxiv.org/abs/1212.0467).
  - I reproduce the Algorithm 2(AltMinComplete) only. Because, Algorithm 1 is not for matrix completion problem, but for matrix sensing problem.
- How to init and run
```
virtualenv .venv -p python3
. .venv/bin/activate
pip install -r requirements.txt
python run.py
```
