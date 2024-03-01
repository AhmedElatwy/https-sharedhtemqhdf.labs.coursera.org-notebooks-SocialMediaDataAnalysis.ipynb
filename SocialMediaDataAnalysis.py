#!/usr/bin/env python
# coding: utf-8

# # Clean & Analyze Social Media

# ## Introduction
# 
# Social media has become a ubiquitous part of modern life, with platforms such as Instagram, Twitter, and Facebook serving as essential communication channels. Social media data sets are vast and complex, making analysis a challenging task for businesses and researchers alike. In this project, we explore a simulated social media, for example Tweets, data set to understand trends in likes across different categories.
# 
# ## Prerequisites
# 
# To follow along with this project, you should have a basic understanding of Python programming and data analysis concepts. In addition, you may want to use the following packages in your Python environment:
# 
# - pandas
# - Matplotlib
# - ...
# 
# These packages should already be installed in Coursera's Jupyter Notebook environment, however if you'd like to install additional packages that are not included in this environment or are working off platform you can install additional packages using `!pip install packagename` within a notebook cell such as:
# 
# - `!pip install pandas`
# - `!pip install matplotlib`
# 
# ## Project Scope
# 
# The objective of this project is to analyze tweets (or other social media data) and gain insights into user engagement. We will explore the data set using visualization techniques to understand the distribution of likes across different categories. Finally, we will analyze the data to draw conclusions about the most popular categories and the overall engagement on the platform.
# 
# ## Step 1: Importing Required Libraries
# 
# As the name suggests, the first step is to import all the necessary libraries that will be used in the project. In this case, we need pandas, numpy, matplotlib, seaborn, and random libraries.
# 
# Pandas is a library used for data manipulation and analysis. Numpy is a library used for numerical computations. Matplotlib is a library used for data visualization. Seaborn is a library used for statistical data visualization. Random is a library used to generate random numbers.

# In[2]:


# your code here
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random


# In[3]:


categories = ['Food', 'Travel', 'Fashion', 'Fitness', 'Music', 'Culture', 'Family' , 'Health']


# In[94]:


data = {
    'Date': [date for _ in range(8) for date in pd.date_range('2020-01-01', periods=500)],
    'Category': [category for category in categories for _ in range(500)],
    'Likes': np.random.randint(0, 1500000, size=4000),
    'Posts':np.random.randint(1, 11200, size=4000),
    'Shares':np.random.randint(0, 100, size=4000),
    'Reach': np.random.randint(100, 2000000, size=4000),
}
df = pd.DataFrame(data)


# ## Checking For Duplicated Values

# In[95]:


df[df.duplicated()] # no duplicates


# ### we Can notice the data has no duplicated values

# ## Checking For Null Values

# In[96]:


df.isnull().sum() # No Null Values


# ### We can See The Data Has No Null Values

# ## Checking For Incorrect Data Types

# In[97]:


df.dtypes


# ### All Data Types Are Correct

# ## A Closer Look At The Data

# In[98]:


df.shape


# In[99]:


df.head()


# In[100]:


df[df['Reach'] < df['Likes']].head()


# ### We Notice Reach Amount is Lower Than The Amount of Likes in Entry 4, to fix that we need drop the rows with this incorrect value

# In[101]:


for i in range(0, 4000):
    if df.loc[i, 'Reach'] < df.loc[i, 'Likes']:
        df.drop(i,inplace=True)


# In[102]:


df[df['Reach'] < df['Likes']].head()


# # Now The Data Is Cleaned And Ready To be Analyzed

# ## Creating Proper Groub By Subset of the Dataframe For Better Visualization

# In[49]:


grp = df.groupby('Category')['Likes'].mean().astype(int)


# ## Creating Line Chart To identify The Difference In Average Likes Between Categories

# In[103]:


plt.figure(figsize=(18,9))
grp.plot(kind='line')
plt.xlabel("Categories")
plt.ylabel("Average Likes")
plt.title("Categories By Average Likes")


# ### Family Topic has the highest average likes

# ## Creating Bar plot to identify how Frequent each Category is discussed

# In[105]:


plt.figure(figsize=(18,9))
sns.barplot(data=df,y='Posts',x='Category')


# ### Music Category Has The Highest Amount of Posts Related

# ## Creating Boxplot To Identify The Outliers and the IQR 

# In[106]:


plt.figure(figsize=(18,9))
sns.boxplot(data=df,x="Category",y="Likes")


# ### Creating Seperate Year And Month Columns

# In[134]:


df['Year'] = df['Date'].dt.strftime('%Y').astype(int)
df['Month'] = df['Date'].dt.strftime('%m')


# In[143]:


new_grp = df.pib


# ## Let's Find Out How Each Category Did Well According to Likes And Posts For The Time Provided

# In[150]:


grouped_df = df.groupby('Category')

# Create a plot with separate lines for each category
fig, ax = plt.subplots(figsize=(18, 9))
for category, data in grouped_df:
    sns.lineplot(data=data, x='Year', y='Posts', label=category,ci=None, ax=ax)

# Customize the plot
plt.xlabel("Year")
plt.ylabel("Posts")
plt.title("Categories By Amount of Posts")
plt.legend(title="Category")

# Show the plot
plt.show()


# In[149]:


grouped_df = df.groupby('Category')

# Create a plot with separate lines for each category
fig, ax = plt.subplots(figsize=(18, 9))
for category, data in grouped_df:
    sns.lineplot(data=data, x='Year', y='Likes', label=category,ci=None, ax=ax)

# Customize the plot
plt.xlabel("Year")
plt.ylabel("Likes")
plt.title("Categories By Amount of Likes")
plt.legend(title="Category")

# Show the plot
plt.show()


# ### The Data Has No Outliers

# In[107]:


df.describe()


# ## Checking For Relation Between Numeric Values

# In[108]:


df.corr()


# ### we can see that The Amount of Reach Has A Small Relationship with The Amount Likes

# In[109]:


plt.figure(figsize=(16,9))
sns.regplot(data=df,x='Likes',y="Reach")


# ### As We Can See The Amount of Likes And Reach Have a potential Relationship in higher values

# # let's Try and predict data using linear regression

# In[124]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
X= df[['Posts','Reach']]
Y= df['Likes']
lr.fit(X,Y)
Y_Hat= lr.predict(X)
r2_score = lr.score(X,Y)
r2_score


# ### The R^2 score is so low that means the predictions won't be accurate

# # Conclusion

# 
# ### 1- Music Category Has The Highest Amount of Posts Related
# ### 2- Family Topic has the highest average likes
# ### 3- The Data Has No Relation Between Numeric Values
# ### 4- Culture, Fashion, Music, Food and family had a decrease in the total of likes through the years
# ### 5- Fitness, Travel and health had An Inecrease in the total of likes through the years
# ### 6- Health was the only category to have a decrease in the total amount of posts through the years
# 

# 
# ### Author
# 
# # Ahmed A. Elatwy

# ## Thank You For Your Time
