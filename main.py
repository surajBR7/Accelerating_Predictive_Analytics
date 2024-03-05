import time
import pandas as pd
import multiprocessing as mp
from functools import partial
from model import new_model


def model(data):
    new_model.predict(data.reshape(1,-1))
    return None


if __name__ == '__main__':
    
    df = pd.read_csv('data.csv')

    features = ['num_pages', 'book_rating', 'book_price', 'text_lang']
    target = 'book_genre'

    for j in [1000, 10000, 20000, 50000]:
        
        x = df[features][:j].values
        
        # Serial
        s1 = time.time()
        for i in range(x.shape[0]):
            model(x[i])
        e1 = time.time()

        # Parallel
        s2 = time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            result = pool.map(model, x)
        e2 = time.time()

        speedup = (e1 - s1) / (e2 - s2)
        efficiency = (speedup / mp.cpu_count()) * 100

        print('Time for Serial: {:.4f} s\nTime for Parallel: {:.4f} s\nData size: {}\nSpeedup: {:.2f}\nEfficiency: {:.2f}%\n'.format(e1 - s1, e2 - s2, j, speedup, efficiency))
