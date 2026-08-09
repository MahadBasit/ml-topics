import pandas as pd
pd.set_option('display.max_rows', 5)

categories = pd.read_csv('categories.csv')
customers = pd.read_csv('customers.csv')
employees = pd.read_csv('employees.csv')
order_details = pd.read_csv('order_details.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')

#Revenue per Category
nw = order_details.merge(products, on='ProductID')
nwd = nw.merge(categories, on='CategoryID')
nwd['revenue'] = nwd['UnitPrice_x'] * nwd['Quantity']
print(nwd.groupby('CategoryName')['revenue'].sum().sort_values(ascending=False))

#Orders per Employee
nwd1 = orders.merge(employees, on='EmployeeID')
print(nwd1.groupby('EmployeeID').size().sort_values(ascending=False))
#Revenue per Employee
nwd2 = orders.merge(order_details, on='OrderID')
nwd2['revenue'] = nwd2['UnitPrice'] * nwd2['Quantity']
print(nwd2.groupby('EmployeeID')['revenue'].sum().sort_values(ascending=False))

#Orders per Country
nwd3 = orders.merge(customers, on='CustomerID')
print(nwd3.groupby('Country').size().sort_values(ascending=False))