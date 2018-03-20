# matrix-completion

논문에 나온 Matrix Completion 알고리즘들을 구현해 보는 Repository.

## How to init

1.  먼저 `app/data/raw/`에 movie-lens 데이터 넣어야 함. 데이터는 [여기](https://grouplens.org/datasets/movielens/)에서 받을 수 있음. zip 파일을 받아서, 아래 폴더에 셋팅해야 하며, 아래 폴더들은 이미 .gitignore 에 추가되어 있음.

```
app/data/raw/ml-100k/
app/data/raw/ml-10m/
app/data/raw/ml-1m/
app/data/raw/ml-20m/
```

2.  virtualenv 셋팅 및 필요 패키지 설치.

```
virtualenv .venv -p python3
pip install -r requirements/common.txt
```

3.  `app/configs/local.py`를 초기화 `app/configs/local.py.example`에 예시 존재.

4.  다음 스크립트를 실행시켜, data/processed 에 필요한 데이터 초기화.

```
python init.py
```

## How to run

```
. .venv/bin/activate
python launcher.py
```
