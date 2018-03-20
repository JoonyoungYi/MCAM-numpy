from app.runner import test
from app.matrix_completion.alternating_minimization import AlternatingMinimization

if __name__ == '__main__':
    test(AlternatingMinimization(), 'ml-100k')
