import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Linear Regression/Salary_Data.csv")
x_train = data["YearsExperience"].values
y_train = data["Salary"].values

def cost(x,y,w,b):
  m = len(x)
  cost_sum = 0
  for i in range(m):
    f = w * x[i] + b
    cost = (f - y)**2
    cost_sum += cost
  total_cost = (1/(2*m)) * cost_sum
  return total_cost


def gradient_func(x,y,w,b):
  m = len(x)
  dc_dw = 0
  dc_db = 0
  for i in range(m):
    f = w * x[i] + b
    dc_dw += (f - y[i]) * x[i]
    dc_db += (f - y[i])
  dc_dw *= (1/m)
  dc_db *= (1/m)
  return dc_dw,dc_db

def grad_descent(x,y,alpha,iter):
  w = 0
  b = 0
  for i in range(iter):
    dc_dw, dc_db = gradient_func(x,y,w,b)
    w = w - alpha * dc_dw
    b = b - alpha * dc_db
    print(f"Iteration (i): Cost {cost(x,y,w,b)}")
  return w,b



learning_rate = 0.01
iterations = 10000
final_w,final_b = grad_descent(x_train,y_train,learning_rate,iterations)

plt.scatter(x_train, y_train, label='Data Points')

x_vals = np.linspace(min(x_train), max(x_train), 100)
y_vals = final_w * x_vals + final_b
plt.plot(x_vals, y_vals, color='red', label='Regression Line')

plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.legend()
plt.show()