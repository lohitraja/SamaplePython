import pandas as pd
import numpy as np
df = pd.read_csv("googleplaystore.csv")
df.head(5)

print(pd.isna(df['App']).value_counts())
print(pd.isna(df['Category']).value_counts())
print(pd.isna(df['Rating']).value_counts())
print(pd.isna(df['Reviews']).value_counts())
print(pd.isna(df['Size']).value_counts())
print(pd.isna(df['Installs']).value_counts())
print(pd.isna(df['Type']).value_counts())
print(pd.isna(df['Price']).value_counts())
print(pd.isna(df['Content Rating']).value_counts())
print(pd.isna(df['Genres']).value_counts())
print(pd.isna(df['Last Updated']).value_counts())
print(pd.isna(df['Current Ver']).value_counts())
print(pd.isna(df['Android Ver']).value_counts())

df1 = df.dropna()
df1.reset_index(inplace=True, drop = True)
df1.head(16)

df1.groupby('Size')['App'].nunique()

for i in range(len(df1)):
    if df1['Size'][i] == 'Varies with device':
        df1['Size'][i] = np.nan
df2 = df1.dropna()
df2.reset_index(inplace=True, drop = True)
df2.head(16)


df2['Unit'] = "NA"
for i in range(len(df2)):
    val = df2['Size'][i]
    val_last = val[-1] 
    num_val = val[:-1]
    df2['Size'][i] = num_val
    df2['Unit'][i] = val_last
df2.head(10)

df2["Size"] = pd.to_numeric(df2["Size"], downcast="float")


for i in range(len(df2)):
    if df2['Unit'][i] == "M":
        df2['Size'][i] = df2['Size'][i]*1000

df2.groupby('Size')['App'].nunique()



for i in range(len(df2)):
    val = df2['Installs'][i]
    last_val = val[-1]
    first_val = val[:-1]
    if last_val == '+':
        df2['Installs'][i]=first_val


import re
for i in range(len(df2)):
    val = df2["Installs"][i]
    val_fill = re.sub(",","",val)
    df2["Installs"][i] = val_fill
df2["Installs"] = pd.to_numeric(df2["Installs"], downcast="integer")



for i in range(len(df2)):
    val = df2["Price"][i]
    if val != '0':
        val_fill = val[1:]
        df2["Price"][i] = val_fill
#df2["Price"] = pd.to_numeric(df2["Price"], downcast="float")
df2["Price"].unique()


df2["Price"] = pd.to_numeric(df2["Price"], downcast="float")
df2["Price"].unique()


print(len(df2))
for i in range(len(df2)):
    if int(df2['Reviews'][i]) > int(df2['Installs'][i]):
        df2 = df2.drop([i], axis=0)
print(len(df2))


df2.reset_index(inplace=True, drop = True)
print(len(df2))
for i in range(len(df2)):
    if (df2['Type'][i]=='Free') & (df2['Price'][i]>0):
        df2 = df2.drop([i], axis=0)
print(len(df2))


df2['Rating'].unique()

df_Price = pd.DataFrame(df2['Price'])
df_Price.plot.box(grid='True')

df2["Reviews"] = pd.to_numeric(df2["Reviews"], downcast="integer")
df_reviews = pd.DataFrame(df2['Reviews'])
df_reviews.plot.box(grid='True')

df_Rating = pd.DataFrame(df2['Rating'])
df_Rating.plot.box(grid='True')
df_Rating.hist()

df_Size = pd.DataFrame(df2['Size'])
df_Size.plot.box(grid='True')
df_Size.hist()


ax1 = df2.plot(kind='scatter', x='Rating', y='Size', color='r') 
ax2 = df2.plot(kind='scatter', x='Rating', y='Reviews', color='g')
ax2 = df2.plot(kind='scatter', x='Rating', y='Price', color='b')
#ax2 = df.plot(kind='scatter', x='Size', y='d', color='g', ax=ax1)   


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,8))
plt.suptitle('')
df2.boxplot(column=['Rating'], by='Content Rating', ax=ax)


fig, ax = plt.subplots(figsize=(20,10))
plt.suptitle('')
df2.boxplot(column=['Rating'], by='Category', ax=ax)
plt.xticks(rotation=90)


print(len(df2))
for i in range(len(df2)):
    if (df2['Price'][i]>200):
        df2 = df2.drop([i], axis=0)
print(len(df2))
df2.reset_index(inplace=True, drop = True)

print(len(df2))
for i in range(len(df2)):
    if (df2['Reviews'][i]>2000000):
        df2 = df2.drop([i], axis=0)
print(len(df2))
df2.reset_index(inplace=True, drop = True)


df_Installs = pd.DataFrame(df2['Installs'])
df_Installs.plot.box(grid='True')
df_new = pd.DataFrame(df_Installs['Installs'].value_counts())
df_new


print(len(df2))
for i in range(len(df2)):
    if (df2['Installs'][i]<50) or (df2['Installs'][i]>100000001):
        df2 = df2.drop([i], axis=0)
print(len(df2))
df2.reset_index(inplace=True, drop = True)


df2.head(5)

df3 = df2.drop(columns=['App', 'Last Updated','Current Ver','Android Ver','Unit'])
df3.head(5)


df3['log_Reviews'] = np.log(df3['Reviews'])
df3['log_Installs'] = np.log(df3['Installs'])
df3.head(5)

print(df3['Category'].unique())
print()
print(df3['Genres'].unique())
print()
print(df3['Content Rating'].unique())


df4 = df3
X = {'Everyone':'10', 'Teen':'20', 'Everyone 10+':'30', 'Mature 17+':'40', 'Adults only 18+':'50', 'Unrated':'60'}
df4['Content Rating'] = df4['Content Rating'].map(X)
df4.head(5)


Y = {'ART_AND_DESIGN':'10','AUTO_AND_VEHICLES':'20','BEAUTY':'30','BOOKS_AND_REFERENCE':'40','BUSINESS':'50','COMICS':'60',
'COMMUNICATION':'70','DATING':'80','EDUCATION':'90','ENTERTAINMENT':'100','EVENTS':'110','FINANCE':'120','FOOD_AND_DRINK':'130',
'HEALTH_AND_FITNESS':'140','HOUSE_AND_HOME':'150','LIBRARIES_AND_DEMO':'160','LIFESTYLE':'170','GAME':'180','FAMILY':'190',
'MEDICAL':'200','SOCIAL':'210','SHOPPING':'220','PHOTOGRAPHY':'230','SPORTS':'240','TRAVEL_AND_LOCAL':'250','TOOLS':'260',
'PERSONALIZATION':'270','PRODUCTIVITY':'280','PARENTING':'290','WEATHER':'300','VIDEO_PLAYERS':'310','NEWS_AND_MAGAZINES':'320',
'MAPS_AND_NAVIGATION':'330'
}
df4['Category'] = df4['Category'].map(Y)
df4.head(5)


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df4["Genres"] = lb_make.fit_transform(df4["Genres"])


df4 = df4.drop(columns = ['Type','Reviews','Installs'])
df4.head(10)


corrMatrix = df4.corr()
print (corrMatrix)


import seaborn as sn
import matplotlib.pyplot as plt
sn.heatmap(corrMatrix, annot=True)
plt.show()


from sklearn.model_selection import train_test_split
X = df4[['Category','log_Reviews', 'Size','log_Installs', 'Price','Content Rating', 'Genres']].values
y = df4['Rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


from sklearn.linear_model import LinearRegression
from sklearn import metrics
regressor = LinearRegression() 
regressor.fit(X_train, y_train)


print(regressor.intercept_) #To retrieve the intercept:
print(regressor.coef_) 


y_pred = regressor.predict(X_test)
df5 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df5

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




