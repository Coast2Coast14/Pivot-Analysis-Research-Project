# Data on Google Sheets index 22 does not load in, gets error
# ("This XML file does not appear to have any style information associated with it. The document tree is shown below."
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc

path = "/Users/elijahwooten/Desktop/Pivot Analysis Project/"
'''
f = []
for dirpath, dirnames, filenames in os.walk(path):
    f = filenames
    break

for files in f:
    if "~" in files:
        f.remove(files)

f.remove(".DS_Store")

# creates empty dataframe
df = pd.DataFrame()
for files in f:
    print(files)
    df = df.append(pd.read_excel(path + files, header=1))

df.to_csv('/Users/elijahwooten/PycharmProjects/Pivot_Analysis/Pivot Data.csv')
'''
'''
my variables used:
position
3pt percentage
midrange percentage
player_usage
assist_percentage
midrange percentage
at the rim percentage
free throw rate
'''

df = pd.read_csv('/Users/elijahwooten/Desktop/Pivot Data/Pivot Data.csv')

mask = ((df['player_profile_three_attempts'] >= 25) & (df['player_profile_mid_attempts'] >= 25) & (df['player_boxscore_fga'] >= 50) & (df['player_boxscore_ast'] >= 10)
        & (df['player_profile_atr_attempts'] >= 25) & (df['position'] == 'G') & (df['player_boxscore_minutes'] >= 100))

my_df_small = df.loc[mask]

my_df_small = my_df_small[['position', 'player_shooting_efficiency_fg3_perc', 'player_profile_mid_percentage', 'player_usage', 'player_assists_assist_percentage',
                           'player_profile_atr_percentage', 'player_four_factors_ftr', 'player_profile_assisted_percentage', 'player_boxscore_stl']]
my_df_small = my_df_small.dropna()
my_df_small = my_df_small[my_df_small['position'] != '*']
my_df_small = my_df_small.drop('position', axis = 1)

my_df_scaled = normalize(my_df_small)
my_df_scaled = pd.DataFrame(my_df_scaled, columns = my_df_small.columns)

plt.figure(figsize=(10,7))
plt.title("Dends")
my_dend = shc.dendrogram(shc.linkage(my_df_scaled, method = 'ward'))
plt.axhline(y=.25, color='r', linestyle='--')
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
test = cluster.fit_predict(my_df_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(my_df_small['player_shooting_efficiency_fg3_perc'], my_df_small['player_profile_assisted_percentage'], c=cluster.labels_)
plt.show()

my_df_small['cluster'] = test

# Gets number of players in each cluster
my_df_small['cluster'].value_counts()

x = my_df_small.groupby('cluster')

x.mean().round(4).to_csv('Cluster_Data.csv')
