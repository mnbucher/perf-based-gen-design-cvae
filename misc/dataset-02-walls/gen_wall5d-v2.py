import numpy as np
import pandas as pd
import csv
from pycel import ExcelCompiler
from skopt.space import Space
from skopt.sampler import Sobol
import pdb

# x1: NQk
# x2: t
# x3: b
# x4: hS
# x5: fk

n_samples = 2**11

sobol = Sobol()
space = Space([(40., 100.),(100., 250.),(2., 3.5)])
x1_x3_x4 = sobol.generate(space.dimensions, n_samples)
x1_x3_x4 = np.array(x1_x3_x4)

x2_t = [11.5, 17.5, 24, 32.4, 36.5]
x5_fk = [2.7, 3.1, 3.5, 3.9, 4.6, 5.3]

n_total = x1_x3_x4.shape[0]*len(x2_t)*len(x5_fk)

print(f"generating dataset of size: {n_total}")

#fn = "./misc/wall-data/mw_ec6_master-tweak.xlsx"
fn = "./misc/dataset-02-walls/mw_ec6_master-tweak.xlsx"

excel = ExcelCompiler(filename=fn)


# generate data

x_all = np.zeros((n_total, 5))
y_all = np.zeros((n_total, 4))

row_idx = 0

for row_x1_x3_x4 in x1_x3_x4:
	for x2 in x2_t:
		for x5 in x5_fk:
			x1 = row_x1_x3_x4[0]
			x3 = row_x1_x3_x4[1]
			x4 = row_x1_x3_x4[2]
			x_all[row_idx, :] = np.array([x1, x2, x3, x4, x5])

			excel.evaluate('1!C22')
			excel.evaluate('1!C23')
			excel.evaluate('1!C25')
			excel.evaluate('1!C26')
			excel.evaluate('1!C27')
			excel.evaluate('1!F46')
			excel.evaluate('1!C40')
			excel.evaluate('1!F40')
			excel.evaluate('1!B49')

			excel.set_value('1!C22', x1)
			excel.set_value('1!C23', x2)
			excel.set_value('1!C25', x3)
			excel.set_value('1!C26', x4)
			excel.set_value('1!C27', x5)

			y1 = excel.evaluate('1!F46') # eta_w
			y2 = excel.evaluate('1!C40') # phi_1
			y3 = excel.evaluate('1!F40') # phi_2
			y4 = 1 if excel.evaluate('1!B49') == "OK" else 0 # schlankheit

			y_all[row_idx, :] = np.array([y1, y2, y3, y4])

			row_idx += 1

df = pd.DataFrame(np.concatenate((x_all, y_all), axis=1), columns=[ f"x{i}" for i in range(x_all.shape[1]) ] + [ f"y{i}" for i in range(y_all.shape[1]) ])
df.to_csv("./misc/dataset-02-walls/5d-wall-v2.csv", index=False)

#np.savetxt("./data/train-test-split/x/x-5d-wall-v2.csv", x_all, delimiter=',')
#np.savetxt("./data/train-test-split/y/y-5d-wall-v2.csv", y_all, delimiter=',')
