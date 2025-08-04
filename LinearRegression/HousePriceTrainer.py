from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing


housing = fetch_california_housing()
for i in housing:
    print(i)
    if(i=='data'):
        for j in housing.__getitem__(i):
            print(j)
    else:
        print(housing.__getitem__(i))

# print(housing)

x1=housing.__getitem__('data')
print(x1)
y=housing.__getitem__('target')
print(y)

x=x1[:,[0]]
print(x)

# model=LinearRegression()
# model.fit(x,y)

# print(model.predict([[0],[1],[5],[10],[25],[50],[75],[100]]))

