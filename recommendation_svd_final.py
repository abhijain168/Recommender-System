
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os


# In[6]:

os.chdir('C:/Users/abhij/Documents/Academics/Recommendation System/ml-1m')


# In[7]:

train=pd.read_csv("ratings.dat",sep="::",header=None)
train.columns = ['user','movie','rating','timestamp']


# In[8]:

count_movie = train.groupby('movie').count()
count_movie = count_movie.drop('rating',axis = 1)
count_movie = count_movie.drop('timestamp',axis = 1).reset_index()
movie_rated_more_than_10 = count_movie[count_movie["user"]>10]
data_temp = pd.merge(movie_rated_more_than_10, train, how='left', on='movie')
data_temp = data_temp.drop('user_x',1)
data_temp.columns = ['movie','user','rating','timestamp']


# In[9]:

data_temp.head()


# In[55]:

data_temp['bool_ratings'] = 0
data_temp['bool_ratings'] = np.where(data_temp['rating']>0,1,0)


# In[10]:

import random
random.seed(23)
from sklearn.model_selection import train_test_split


# In[11]:

x_train, x_test, y_train, y_test = train_test_split(data_temp.iloc[:,:2], data_temp.iloc[:,2], test_size=0.20, random_state=42)


# In[12]:

x_train['rating'] = y_train


# In[13]:

train_rating = x_train.pivot(index = 'user', columns = 'movie', values = 'rating')


# In[14]:

train_rating.shape


# In[15]:

train_rating = train_rating.fillna(0)


# In[16]:

train_rating = train_rating.as_matrix()


# In[17]:

train_rating


# In[18]:

len(x_test.movie.unique())


# In[19]:

from numpy.linalg import solve


# In[20]:

n_users, n_items = train_rating.shape
n_factors = 10
item_reg = 0
user_reg = 0


# In[21]:

#user_vecs = np.random.random((n_users, n_factors))
#item_vecs = np.random.random((n_items, n_factors))


# In[22]:

item_bias_reg = 0
user_bias_reg = 0
item_fact_reg = 0
user_fact_reg = 0


# In[23]:

# initialize latent vectors        
user_vecs = np.random.normal(scale=1./n_factors,size=(n_users, n_factors))
item_vecs = np.random.normal(scale=1./n_factors,size=(n_items, n_factors))


# In[24]:

learning_rate = 0.01
user_bias = np.zeros(n_users)
item_bias = np.zeros(n_items)
global_bias = np.mean(train_rating[np.where(train_rating != 0)])


# In[25]:

global_bias


# In[26]:

sample_row, sample_col = train_rating.nonzero()
n_samples = len(sample_row)


# In[27]:

training_indices = np.arange(n_samples)
np.random.shuffle(training_indices)


# In[69]:

#training_indices = np.array([162741])


# In[70]:

#training_indices


# In[32]:

n_iter = 40


# In[29]:

def predict( u, i):
    """ Single user and item prediction."""
    prediction = global_bias + user_bias[u] + item_bias[i]
    prediction += user_vecs[u, :].dot(item_vecs[i, :].T)
    return prediction


# In[30]:

import time


# In[31]:

mse_train = []


# In[35]:

item_vecs[i, :].shape


# In[33]:

start_time = time.time()
ctr = 1
while ctr<= n_iter:
    training_indices = np.arange(n_samples)
    np.random.shuffle(training_indices)
    error_sum = 0
    for idx in training_indices:
        u = sample_row[idx]
        i = sample_col[idx]
        prediction = predict(u,i)
        e = train_rating[u,i] - prediction
        error_sum = error_sum + e*e
        
        #update biases
        user_bias[u] = user_bias[u] + learning_rate*(e - user_bias_reg* user_bias[u])
        item_bias[i] = item_bias[i] + learning_rate*(e - item_bias_reg* item_bias[i])
    
        #update latent factors:
        user_vecs[u, :] += learning_rate * (e * item_vecs[i, :] - user_fact_reg * user_vecs[u,:])
        item_vecs[i, :] += learning_rate * (e * user_vecs[u, :] - item_fact_reg * item_vecs[i,:])
        
    rmse = error_sum/n_samples
    mse_train.append(error_sum)
    print("rmse for iteration ",ctr," is ", rmse)
    ctr = ctr + 1

print("total time taken: ",time.time()-start_time)


# In[36]:

import matplotlib.pyplot as plt


# In[37]:

plt.plot(mse_train)
plt.show()


# # For test data:

# In[38]:

raw_train_rating = x_train.pivot(index = 'user', columns = 'movie', values = 'rating')


# In[39]:

raw_train_rating.head()


# In[40]:

x_test.head()


# In[41]:

u = raw_train_rating.index.get_loc(5)
i = raw_train_rating.columns.get_loc(6)


# In[42]:

prediction = predict(u,i)


# In[43]:

prediction


# In[44]:

x_test['actual_rating'] = y_test


# In[45]:

x_test['predicted_rating'] = 0


# In[46]:

x_test.head()


# In[47]:

error_sum = 0


# In[48]:

for j in range(x_test.shape[0]):
    u  = raw_train_rating.index.get_loc(x_test.iloc[j,1])
    i  = raw_train_rating.columns.get_loc(x_test.iloc[j,0])
    x_test.iloc[j,3] = predict(u,i)
    error = x_test.iloc[j,2] - x_test.iloc[j,3]
    error_sum = error_sum + error*error
    


# In[50]:

rms = np.sqrt(error_sum/x_test.shape[0])


# In[51]:

rms


# In[93]:

from sklearn.metrics import mean_squared_error
from math import sqrt



# In[94]:

test_rmse = sqrt(mean_squared_error(x_test['actual_rating'], x_test['predicted_rating']))


# In[95]:

test_rmse


# # Grid search for different parameters:

# In[96]:

iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
regularizations = [0.001, 0.01, 0.1, 1.]


# In[97]:

n_users, n_items = train_rating.shape
global_bias = np.mean(train_rating[np.where(train_rating != 0)])
sample_row, sample_col = train_rating.nonzero()
n_samples = len(sample_row)
training_indices = np.arange(n_samples)
np.random.shuffle(training_indices)


# In[98]:

n_iter = 40


# In[52]:

for reg in regularizations:
    item_bias_reg = reg
    user_bias_reg = reg
    item_fact_reg = reg
    user_fact_reg = reg
    for learning_rate in regularizations:
        # initialize latent vectors        
        user_vecs = np.random.normal(scale=1./n_factors,size=(n_users, n_factors))
        item_vecs = np.random.normal(scale=1./n_factors,size=(n_items, n_factors))
        user_bias = np.zeros(n_users)
        item_bias = np.zeros(n_items)

        print("\nlatent factors = ",n_factors)
        print("\nregularisation = ",reg)
        print("\nlearning rate = ",learning_rate)
        start_time = time.time()
        ctr = 1
        while ctr<= n_iter:
            training_indices = np.arange(n_samples)
            np.random.shuffle(training_indices)
            error_sum = 0
            for idx in training_indices:
                u = sample_row[idx]
                i = sample_col[idx]
                prediction = predict(u,i)
                e = train_rating[u,i] - prediction
                error_sum = error_sum + e*e
                #update biases
                user_bias[u] = user_bias[u] + learning_rate*(e - user_bias_reg* user_bias[u])
                item_bias[i] = item_bias[i] + learning_rate*(e - item_bias_reg* item_bias[i])

                #update latent factors:
                user_vecs[u, :] += learning_rate * (e * item_vecs[i, :] - user_fact_reg * user_vecs[u,:])
                item_vecs[i, :] += learning_rate * (e * user_vecs[u, :] - item_fact_reg * item_vecs[i,:])

        rmse = error_sum/n_samples
        #mse_train.append(error_sum)
        #print("rmse for iteration ",ctr," is ", rmse)
        ctr = ctr + 1

        print("\ntotal time taken: ",time.time()-start_time)
        print("\ntrain rmse:", rmse)

        for j in range(x_test.shape[0]):
            us  = raw_train_rating.index.get_loc(x_test.iloc[j,1])
            it  = raw_train_rating.columns.get_loc(x_test.iloc[j,0])
            x_test.iloc[j,3] = predict(us,it)

        test_rmse = sqrt(mean_squared_error(x_test['actual_rating'], x_test['predicted_rating']))

        print("\ntest rmse: ",test_rmse)            


# In[ ]:



