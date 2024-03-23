import pandas as pd
import numpy as np


df = pd.read_csv("220427_KI-Futter.csv", delimiter=',', index_col=None)

print(df)

#: Features X: wurden benutzt, um die Bilder der Räume zu erzeugen
x_col_names = ['LY-A','LY-B','SC-A','SC-B','OP']

# Targets Y: das soll der Nutzer eingeben können (wurden in der Umfrage von Befragten eingefüllt)
y_col_names = ['F-PLE','F-STI','F-COM','F-DYN','F-PUB']


df_filtered = df[x_col_names + y_col_names]

df_filtered.columns = [f"x{i}" for i in range(len(x_col_names))] + [f"y{i}" for i in range(len(y_col_names))]

df_filtered = df_filtered.replace({0.0: 1.0, 0.25: 1.25, 0.5: 1.5, 0.75: 1.75, 1.0: 2.0})

print(df_filtered)

df_filtered.to_csv('rooms_xy_v3_shifted.csv', index=False)