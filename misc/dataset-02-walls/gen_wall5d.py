import numpy as np
import pandas as pd
import csv
from pycel import ExcelCompiler


# define parameter ranges

# NQk
x1_nqk = list(np.linspace(40, 100, 13))
# t
x2_t = [11.5, 17.5, 24, 32.4, 36.5]
# b
x3_b = list(np.linspace(100, 250, 7))
# hS
x4_hS = list(np.linspace(2, 3.5, 7))
# fk
x5_fk = [2.7, 3.1, 3.5, 3.9, 4.6, 5.3]

n_total = len(x1_nqk)*len(x2_t)*len(x3_b)*len(x4_hS)*len(x5_fk)

print(f"generating dataset of size: {n_total}")
fn = "./data-2-wall/mw_ec6_master-tweak.xlsx"
excel = ExcelCompiler(filename=fn)


# generate data

x_all = np.zeros((n_total, 5))
y_all = np.zeros((n_total, 4))

row_idx = 0

for x1 in x1_nqk:
	for x2 in x2_t:
		for x3 in x3_b:
			for x4 in x4_hS:
				for x5 in x5_fk:
					x_all[row_idx, :] = np.array([x1, x2, x3, x4, x5])

					excel.evaluate('1!C22')
					excel.evaluate('1!C23')
					excel.evaluate('1!C25')
					excel.evaluate('1!C26')
					excel.evaluate('1!C27')

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


np.savetxt("./data-2-wall/x/x-5d-wall.csv", x_all, delimiter=',')
np.savetxt("./data-2-wall/y/y-5d-wall.csv", y_all, delimiter=',')
