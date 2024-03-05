import csv
import numpy as np
import os
from core.api import read_files, tabulate, write_csv


def export():
    # create a list of dictionries with the read data
    data = read_files("files")

    # convert image series into tabular velocity field
    table = tabulate(data)

    # export data as csv
    write_csv(table, "output.csv")
