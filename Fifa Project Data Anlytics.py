#!/usr/bin/env python
# coding: utf-8

# # Importing Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split


# # Business Understanding
# Being a football fan and local famous striker means exploring FIFA19 player datsset could be so much fun.
# 
# I will focus on the three question below:
# 
# Q1: What's the ratio of total wages/ total potential for clubs. Which clubs are the most economical ？
# 
# Q2: What's the age distribution like? How is it related to player's overall rating?
# 
# Q3: How is a player's skils set influence his potential? Can we predict a player's potential based on his skills' set?
# 
# 

# # Loading Dataset
# 

# In[2]:


df_players= pd.read_csv('data.csv')


# In[3]:


df_players.head(10)


# In[4]:


df_players.info()


# In[5]:


df_players.columns


# In[6]:


df_players.describe()


# In[7]:


df_players.isnull().sum()


# # Cleaning The Data

# In[8]:


sns.heatmap(df_players.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[9]:


df_players.info()


# # 3. Prepare Data
# There are some necessary stpes to apply before continue exploring the dataset:
# 
# Drop unused columns
# 
# Convert string values to number
# 
# Handle missing values, drop them if necessary

# In[10]:


columns_to_drop = ['Unnamed: 0', 'ID', 'Photo', 'Flag','Club Logo', 'Preferred Foot', 
                   'Body Type', 'Real Face', 'Jersey Number', 'Joined', 'Loaned From',
                   'Contract Valid Until', 'Height', 'Weight','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
                   'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
                   'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Release Clause']


# In[11]:


df_players.drop(columns_to_drop,axis=1,inplace=True)


# In[12]:


df_players.head()


# In[13]:


df_players.info()


# In[14]:


def string_to_number(amount):
   
    if amount[-1] == 'M':
        return float(amount[1:-1])*1000000
    elif amount[-1] == 'K':
        return float(amount[1:-1])*1000
    else:
        return float(amount[1:])


# In[15]:


df_players['Value_M'] = df_players['Value'].apply(lambda x: string_to_number(x)/1000000)
df_players['Wages_K'] = df_players['Wage'].apply(lambda x :string_to_number(x)/1000)


# In[16]:


df_players.drop(['Value','Wage'],axis=1,inplace=True)


# In[17]:


df_players.head()


# In[18]:


df_players.describe()


# In[19]:


sns.heatmap(df_players.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[20]:


df_missing_players = df_players[df_players['Curve'].isnull()]


# In[21]:


df_missing_players.sample(10)


# In[22]:


df_missing_players.describe()


# We can see that quite a few columns which are related to players' skills got 48 missing values.
# 
# So there were 48 players that simply missing these values.
# 
# But we will reserve those players for Q1 and Q2 since there were no missing value in Value_M and Wage_K column.
# 
# For Q3, we will drop those player rows since there were just too many missing values here.

# # DataAnalytics
# 

# ## Ratio of total wages / total potential for clubs(which clubs are most economical?)

# In[23]:


club_wages = df_players.groupby('Club').sum()


# In[24]:


club_wages


# In[25]:


club_player_count = df_players.groupby("Club").count()


# In[26]:


club_player_count


# In[27]:


#Number of clubs and avarage number of players in each club
print('Total Number of Club is {}'.format(club_player_count.shape[0]))
print('Avg Number of players in each club is {}'.format(round(club_player_count['Name'].mean(),3)))
print('Total Average Wage(k) and Potential Ratio is {}'.format(round(club_wages['Wages_K'].sum() / club_wages['Potential'].sum(),2)))


# In[28]:


# Finding this details for all clubs
club_wages['Wage/Potential'] = club_wages['Wages_K'] / club_wages['Potential']
club_wages['Player_Number'] = club_player_count['Name'] 
club_wages['Player Avg Age']= club_wages['Age'] / club_wages['Player_Number']


# In[29]:


club_wages.sort_values('Wage/Potential',ascending=False, inplace=True)
club_wages.head()


# In[30]:


club_wages['Wage/Potential'].head(10).plot(kind='bar', color='Blue')
plt.title('Top 10 clubs spending wage(K) on players potential')


# In[31]:


club_wages['Wage/Potential'].tail(10).plot(kind='bar', color='red')
plt.title('Top 10 economical clubs ')


# From the result and plot, it's obvious that the 'Giant' clubs including Real Madrid, Bacelona, and clubs from EPL are willing to spend much more wage for high potential players than average clubs. This is how they stay competitive in leagues.
# 
# But surprisingly, the economical clubs are not clubs from nowhere that we never heard of. Some of them are even quite famous like AEK Athens, Dynamo Kyiv. This suggests that those clubs' players are potiential but underpayed. It maybe a good approach for 'Giant' clubs to import more econimical players from them to reduce their overall wage spent.

# ## Age Distribution and how its related to Players Overall Rating

# In[32]:


age_count = df_players ['Age'].value_counts()
age_count.sort_index(ascending =True,inplace=True)


# In[33]:


age_count.head()


# In[34]:


age_count_list = age_count.values.tolist()
age_mean = df_players.groupby('Age').mean()
age_overall_rating_list = age_mean['Overall'].values.tolist()


# In[35]:


ages = age_count.index.values.tolist()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(ages,age_overall_rating_list, color = 'red', label='Average Rating')
ax1.legend(loc=1)
ax1.set_ylabel('Average Rating')

ax2 = ax1.twinx()
plt.bar(ages, age_count_list, label='Age Count',color='green')
ax2.legend(loc=2)
ax2.set_ylabel('Age Count')
plt.show()


# From above plot, we can see that most players are between 20-26 years old. And players' number start to decrease after 26 years old and speed up after 30. Reason behind this could be that many young player didn't get enough opportunities to prove themselves and give up their dream as a football player.
# 
# When a football player reaches their late 20s, they have gain enough experience and reaches peak of their rating. The golden era of a football player starts here and ends when his age reaches 35. At this age, his physical body condition drops quickly so as average rating.
# 
# There are also quite a few numbers of players with age over 37, 38 years old. This is quite a surprise especially their rating still can remain quite high.

# # Data Analytics with ML

# In[36]:


columns_to_drop = ['Name', 'Nationality', 'Club']
df_players.drop(columns_to_drop, axis=1, inplace=True)


# In[37]:


df_players.dropna(axis=0,how ='any',inplace=True)


# In[38]:


df_players


# In[39]:


df_players['Work Rate Attack'] = df_players['Work Rate'].map(lambda x: x.split('/')[0])
df_players['Work Rate Defence'] = df_players['Work Rate'].map(lambda x: x.split('/')[1])
df_players.drop('Work Rate', axis=1, inplace=True)


# In[40]:


df_players.head()


# In[41]:


# One Hot Encoding for Position, Work Rate Attack, Work Rate Defence
one_hot_columns = ['Position', 'Work Rate Attack', 'Work Rate Defence']
df_players = pd.get_dummies(df_players, columns=one_hot_columns, prefix = one_hot_columns).


# In[42]:


print(df_players.head())

df_players.shape


# # Train Model and Measure Performance

# In[48]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()


# In[49]:


y = df_players['Potential']
X = df_players.drop(['Value_M', 'Wages_K', 'Potential', 'Overall'], axis=1)


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)


# In[51]:


dtree.fit(X_train,y_train)


# ### Predictions and evaluation using DecisionTree

# In[52]:


predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix


# In[53]:


print(classification_report(y_test,predictions))


# In[54]:


print(confusion_matrix(y_test,predictions))


# # So we can't use decision tree in predictions we have to use Random Forrest

# In[42]:


y = df_players['Potential']
X = df_players.drop(['Value_M', 'Wages_K', 'Potential', 'Overall'], axis=1)


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)


# In[44]:


from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.ensemble import RandomForestRegressor


# In[45]:


ForestRegressor = RandomForestRegressor(n_estimators=500)
ForestRegressor.fit(X_train, y_train)
y_test_preds = ForestRegressor.predict(X_test)
print(r2_score(y_test, y_test_preds))
print(mean_squared_error(y_test, y_test_preds))


# In[46]:


coefs_df = pd.DataFrame()

coefs_df['Features'] = X_train.columns
coefs_df['Coefs'] = ForestRegressor.feature_importances_
coefs_df.sort_values('Coefs', ascending=False).head(10)


# Ball control, reactions, and age are the main three features that decides a player's potential. This is same to our perception.
# 
# Young players with excellent ball control and fast reactions tends to give us an outstanding performance in football match.

# In[47]:


coefs_df.set_index('Features', inplace=True)
coefs_df.sort_values('Coefs', ascending=False).head(5).plot(kind='bar', color='blue')


# In[ ]:




