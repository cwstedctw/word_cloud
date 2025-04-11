
from help_function import read_coldata, take_wordCounts, make_removedFile, make_synonymFile
from config import data_file
import pandas as pd

df = pd.read_excel(data_file)
for col in range(20, 27):
   data = read_coldata(df, col)
   word_counts = take_wordCounts(data)
   make_removedFile(word_counts.copy(), col)
   make_synonymFile(word_counts.copy(), col)
