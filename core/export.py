import os
import time
from core.api import read_files, write_csv
import multiprocessing
from functools import partial


def export():
    # create a list of dictionries with the read data
    data = read_files("files")
    timeframes = sorted(set(item["time"] for item in data))

    # start timer
    start_time = time.time()

    # multiprocessing
    partial_worker = partial(write_csv, data)
    pool = multiprocessing.Pool()
    pool.map(partial_worker, timeframes)
    pool.close()
    pool.join()

    # end timer
    end_time = time.time()

    # export csv files
    # for index, timeframe in enumerate(timeframes):
    #     write_csv(data, timeframe)

    execution_time = end_time - start_time
    print("Execution time: {:.2f} seconds".format(execution_time))
