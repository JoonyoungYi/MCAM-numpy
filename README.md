# matrix-completion

논문에 나온 Matrix Completion 알고리즘들을 구현해 보는 Repository

## How to init
1. 먼저 `files/`에 movie-lens 데이터 넣어야 함. 데이터는 [여기](https://grouplens.org/datasets/movielens/)에서 받을 수 있음. .zip파일을 받아서, 아래 폴더에 셋팅해야 하며, 아래 폴더들은 이미 .gitignore에 추가되어 있음.
```
files/ml-100k/
files/ml-10m/
files/ml-1m/
files/ml-20m/
```

2. virtualenv 셋팅
```
virtualenv .venv -p python3
pip install -r requirements/common.txt
```

## How to run
```
python launcher.py
```
