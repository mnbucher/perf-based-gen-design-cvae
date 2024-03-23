import pandas as pd
import numpy as np

df = pd.read_pickle('Hallen_Rohdaten.pkl')

x_col_names = ["Spannweite", "Stuetzenhoehe", "Dachneigung", "Rahmenabstand", "Design_Last", "Stuetzenprofil", "Riegelprofil", "Voutenprofil", "Voutenprofilfaktor", "Voutenlaengenfaktor"]
y_col_names = ["Stuetzenausnutzung", "Riegelausnutzung"]

df_filtered = df[x_col_names + y_col_names]

df_filtered.columns = [f"x{i}" for i in range(len(x_col_names))] + [f"y{i}" for i in range(len(y_col_names))]

print(df_filtered)

df_filtered.to_csv('hallen_xy_v2.csv', index=False)