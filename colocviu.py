import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# Load the dataset
# song_data = np.load("song_data.npy")
# user_data = np.load("user_data.npy")

# 11
# Create the user and music matrix

df = pd.read_csv('user.csv')

pivot_table =  df.pivot_table(values='Replays', index='UserId', columns='MusicId', fill_value=0)
pivot_table= pivot_table.astype(int)
print(pivot_table)

# 9/10

# Dataframe for user.csv


# Read files and filter out users and songs with the fewest replays
data = pd.read_csv('user.csv')
data_by_user = data.groupby('UserId').count()
threshold = data_by_user['Replays'].min()
data = data[data.UserId.isin(data_by_user[data_by_user['Replays'] > threshold].index )]

data_by_music = data.groupby('MusicId').count()
threshold = data_by_music['Replays'].min()
data = data[data.MusicId.isin(data_by_music[data_by_music['Replays'] > threshold].index)]

# Save the clean data
data.to_csv('user_curatat.csv', index=False)
data = data.drop_duplicates()
data.to_csv('user_curatat1.csv', index=False)



# Read data from files
original_data = pd.read_csv('user.csv')
cleaned_data = pd.read_csv('user_curatat1.csv')

# Group data by UserId and calculate the number of Replays
original_data_by_name = original_data.groupby('UserId').count()
cleaned_data_by_name = cleaned_data.groupby('UserId').count()

# Draw plots
plt.bar(original_data_by_name.index, original_data_by_name['Replays'])
plt.ylabel('Replays')
plt.xlabel('UserId')
plt.title('Number of replays by user (before cleaning)')
plt.show()

plt.bar(cleaned_data_by_name.index, cleaned_data_by_name['Replays'])
plt.ylabel('Replays')
plt.xlabel('UserId')
plt.title('Number of replays by user (after cleaning)')
plt.show()

original_data_by_music = original_data.groupby('MusicId').count()
cleaned_data_by_music = cleaned_data.groupby('MusicId').count()

plt.bar(original_data_by_music.index, original_data_by_music['Replays'])
plt.ylabel('Replays')
plt.xlabel('MusicId')
plt.title('Number of replays songs (before cleaning)')
plt.show()

plt.bar(cleaned_data_by_music.index, cleaned_data_by_music['Replays'])
plt.ylabel('Replays')
plt.xlabel('MusicId')
plt.title('Number of replays songs (after cleaning)')
plt.show() 

# Singular Value Decomposition
replays_np = pivot_table.to_numpy()
user_replays_mean = np.mean(replays_np, axis=1)
replays_demeaned = replays_np - user_replays_mean.reshape(-1, 1)

k = 29  # Choose an appropriate value for k
U, sigma, Vt = svds(replays_demeaned, k=k)

# Similarities between users and songs
sigma = np.diag(sigma)

all_user_predicted_replays  = ((U @ sigma) @ Vt) + user_replays_mean.reshape (-1, 1)

userDF = pd.DataFrame (all_user_predicted_replays, columns = pivot_table.columns)

# System Efficiency

from sklearn.metrics import mean_squared_error

# Reconstruct the original matrix

predicted_matrix = ((U @ sigma) @ Vt)

# Calculate the error
rmse = mean_squared_error(pivot_table, predicted_matrix)
print("Mean Squared Error: ", rmse) 


# Music Recommendations

def recommend_music (userDF, userID, musicDF, original_replaysDF, num_recommendations):
    
    # Sort the user's predictions
    user_row_number = userID - 1
    sorted_user_prediction = userDF.iloc[ user_row_number].sort_values(ascending = False)

    # Merge user data with song information
    user_data = original_replaysDF[original_replaysDF.UserId == (userID)]
    user_full = (user_data.merge(musicDF, how='left',left_on = 'MusicId', right_on = 'MusicId')
    .sort_values(['Replays'],ascending = False))
    
    # Print how many songs the user has listened to
    print ('User',userID, 'has already listened ', user_full.shape[0], ' music.')

    # Recommend songs with the most unreplayed replays for the user
    recommendation = (musicDF[~musicDF['MusicId'].isin(user_full ['MusicId'])]
    .merge(pd.DataFrame(sorted_user_prediction).reset_index(),how='left', left_on= 'MusicId', 
    right_on ='MusicId').rename(columns={user_row_number: 'Replays'})
    .sort_values ('Replays', ascending = False).iloc[:num_recommendations, :-1])

    return user_full, recommendation



# Create or read userDF, musicDF, and original_replaysDF DataFrames
userDF = pd.read_csv("user.csv")
musicDF = pd.read_csv("music(1).csv")
original_replaysDF = pd.read_csv("user.csv")

# Call the function
userID = 4
num_recommendations = 4
user, recommendations = recommend_music(userDF, userID, musicDF, original_replaysDF, num_recommendations)

# Print the recommendations
print(recommendations)

