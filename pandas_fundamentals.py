import pandas as pd
pd.set_option('display.max_rows', 5)

table = pd.DataFrame({'Apples': [30], 'Bananas': [21]})
fruits = table
print(fruits)


table = pd.DataFrame({'Apples': [35, 41], 'Bananas': [21, 34]}, index = ['2017 Sales', '2018 Sales'])
fruit_sales = table
print(fruit_sales)


ingredients = pd.Series({'Flour': '4 cups', 'Milk': '1 cup', 'Eggs': '2 large', 'Spam': '1 can'}, name = 'Dinner')
print(ingredients)


#reviews = pd.read_csv('filename.csv', index_col = 0)
#print(reviews)


animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
print(animals)
animals.to_csv('cows_and_goats.csv')