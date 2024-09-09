import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
import numpy as np
import seaborn as sns
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans
 
barplots = []
figures1=[]
heatmaps = []
pca_plots =[]
piecharts = []
umap=[]
tsne_list = []
parallel_coord = []
Kmeans = []
 
# Load the KDD Cup dataset
kddcup_df = pd.read_csv('kddcup.data_10_percent_corrected', header=None)
 
# Display the first few rows to explore the data
 
# Set column names
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]
 
# Assign column names to the DataFrame
kddcup_df.columns = columns
 
# Summary statistics for numerical features
#print(kddcup_df.describe())
 
label_column = kddcup_df.columns[-1]
 
 
# Convert categorical columns using Label Encoding
categorical_columns = ['protocol_type', 'service', 'flag']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    kddcup_df[col] = le.fit_transform(kddcup_df[col].astype(str))
 
# Split the data into features and target
X = kddcup_df.drop(columns=['label'])
y = kddcup_df['label']
 
# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
print(y_train)
X_train = X_train.dropna()
 
# Train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
 
importances = rf.feature_importances_
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
features_df = features_df.sort_values(by='Importance', ascending=False)
 
# Extracting top 10 features (or any number you prefer)
top_features_df = features_df.head(24)
 
# Extracting the names of the top features
top_features_names = top_features_df['Feature'].tolist()
 
# Feature selection

 
print(top_features_names)
 
top_features_dataset = X_train[top_features_names]
top_features_dataset['label']=y_train
 
# Displaying the first few rows of the new dataset with top features
print(top_features_dataset.head())
 
 
 
# Visualize the distribution of attack types in KDD Cup dataset
fig_kddcup_pie = px.pie(
    top_features_dataset,
    names=top_features_dataset['label'],
    title='Distribution of Attack Types in KDD Cup 1999 Dataset'
)
#fig_kddcup_pie.show()
figures1.append(fig_kddcup_pie)
piecharts.append(fig_kddcup_pie)
 
 
 
 
# Calculate class frequencies and percentages
class_counts = top_features_dataset['label'].value_counts()
class_percentages = (class_counts / len(top_features_dataset)) * 100
 
# Create a DataFrame for Plotly Express
data_for_plot = pd.DataFrame({
    'Class': class_counts.index,
    'Frequency': class_counts.values,
    'Percentage': class_percentages.values
})
 
# Create the bar chart
fig1 = px.bar(data_for_plot, x='Class', y='Frequency', text='Percentage')
 
# Adjust the appearance
fig1.update_layout(
    title='Class Distribution of KDD Cup 1999 Dataset',
    xaxis_title='Class',
    yaxis_title='Frequency',
    xaxis={'categoryorder':'total descending'}
)
figures1.append(fig1)
#fig1.show()
barplots.append(fig1)
 
 

# Define scalers
scalers = {
    "Original": None,
    "MinMax": MinMaxScaler(),
    "Quantile": QuantileTransformer()
 
   #  ,"Power": PowerTransformer()
}
 
# Function to calculate Mahalanobis distance
def calculate_mahalanobis(data, labels):
    # Calculate centroids for each class
    centroids = data.groupby(labels).mean()
 
    # Compute covariance matrix
    cov_matrix = np.cov(data.T)
 
    # Regularize covariance matrix to avoid singularity
    regularization_factor = 1e-4
    regularized_cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * regularization_factor
 
    # Compute inverse covariance matrix
    inv_cov_matrix = np.linalg.inv(regularized_cov_matrix)
 
    # Calculate Mahalanobis distance between each pair of centroids
    distances = pd.DataFrame(index=centroids.index, columns=centroids.index)
    for i in centroids.index:
        for j in centroids.index:
            distances.at[i, j] = mahalanobis(centroids.loc[i], centroids.loc[j], inv_cov_matrix)
    return distances
 
# Calculate and plot Mahalanobis distance for each scaler
figures = []  # List to store figures for later display
for name, scaler in scalers.items():
    if scaler:
        # Apply the scaler to the numerical features of kddcup_df
        scaled_features = scaler.fit_transform(top_features_dataset.select_dtypes(include=[np.number]))
    else:
        # Use the original features
        scaled_features = top_features_dataset.select_dtypes(include=[np.number])
 
    # Get numeric features and class labels for Mahalanobis distance calculation
    numeric_features = pd.DataFrame(scaled_features, columns=top_features_dataset.select_dtypes(include=[np.number]).columns)
    class_labels = top_features_dataset['label']
 
    # Calculate Mahalanobis distances
    m_distances = calculate_mahalanobis(numeric_features, class_labels)
 
    # Plot heatmap using Plotly
    fig = px.imshow(m_distances,
                    labels=dict(color="Mahalanobis Distance"),
                    x=m_distances.columns,
                    y=m_distances.index,
                    title=f"Mahalanobis Distance Heatmap - {name} for KDD Cup 1999 Dataset")
    fig.update_xaxes(side="bottom")
    figures1.append(fig)
    heatmaps.append(fig)
    #fig.show()  # Uncomment to show each plot immediately
 
# To display the figures, you can iterate over figures and call .show() on each, or handle as needed
 
 
 
 
# Separate features and labels for PCA
features = top_features_dataset.drop('label', axis=1)
labels = top_features_dataset['label']
# Apply PCA to reduce to three dimensions
 
 
pca_3d_kdd = PCA(n_components=3)
features_pca_3d_kdd = pca_3d_kdd.fit_transform(features)

 
# Create a DataFrame for the PCA results
df_pca_3d_kdd = pd.DataFrame(features_pca_3d_kdd, columns=['PC1', 'PC2', 'PC3'])
df_pca_3d_kdd['label'] = labels.values
 
# Plotting 3D scatter
fig_pca_3d_kdd = px.scatter_3d(
    df_pca_3d_kdd,
    x='PC1',
    y='PC2',
    z='PC3',
    color='label',
    title='3D PCA Visualization of KDD Cup 1999 Dataset'
)
#fig_pca_3d_kdd.show()
 
# Append figure to the list
figures1.append(fig_pca_3d_kdd)
pca_plots.append(fig_pca_3d_kdd)
 
 
 
tsne = TSNE(n_components=2, perplexity=91, n_iter=1000, random_state=42, metric='euclidean')
features_tsne = tsne.fit_transform(features)
 
# Create a DataFrame for the t-SNE results
df_tsne = pd.DataFrame(features_tsne, columns=['Dimension 1', 'Dimension 2'])
df_tsne['label'] = labels.values
 
# Plotting t-SNE visualization for the entire dataset
fig_tsne = px.scatter(
    df_tsne,
    x='Dimension 1',
    y='Dimension 2',
    color='label',
    title='t-SNE Visualization of the KDD Cup 1999 Dataset',
    labels={'color': 'label'}
)
figures1.append(fig_tsne)
fig_tsne.show()
print("tsne")
tsne_list.append(fig_tsne)
 

 
 
 
# Assuming 'top_features_dataset' contains the dataset with top features and 'label'
 
# Select features for the parallel coordinates plot
# Here, I'm using all features in 'top_features_dataset' except 'label'
features_for_parallel_plot = top_features_dataset.drop('label', axis=1)
 
# Create the parallel coordinates plot
fig_parallel = px.parallel_coordinates(
    features_for_parallel_plot,
    color=top_features_dataset['label'].astype('category').cat.codes,  # Encoding labels as colors
    labels={col: col for col in features_for_parallel_plot.columns},  # Customizing labels if needed
    title='Parallel Coordinates Plot for KDD Cup 1999 Dataset'
)
 
# If you want to add this to your Dash app, you can append 'fig_parallel' to one of your figure lists
figures1.append(fig_parallel)  # Or whichever figure list is appropriate
parallel_coord.append(fig_parallel)
 
 

 
 
umap_3d = UMAP(n_components=3, init='random', random_state=0)
 
proj_3d = umap_3d.fit_transform(features)
 
# Convert the projections to DataFrames
df_proj_3d = pd.DataFrame(proj_3d, columns=['PC1', 'PC2', 'PC3'])
 
# Add the 'attack_cat' column (assuming 'attack_cat' is available in your context)
df_proj_3d['label'] = labels.values
 
# Plotting
 
 
fig_3d = px.scatter_3d(
    df_proj_3d, x='PC1', y='PC2', z='PC3',title = 'UMAP for KDD Cup 1999 dataset',
    color='label'
)
 
fig_3d.update_traces(marker_size=5)

figures1.append(fig_3d)
print("umap")
umap.append(fig_3d)
 
 

  
 
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(top_features_dataset.drop('label', axis=1))
 
# Perform PCA to reduce to two dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
 
# Run KMeans clustering
kmeans = KMeans(n_clusters=15, random_state=42)  # Adjust the number of clusters if necessary
kmeans.fit(X_pca)
cluster_centers = kmeans.cluster_centers_
 
# Calculate the pairwise distances between cluster centers
distances = pairwise_distances(cluster_centers)
 
# Determine the radius of the circles to draw around each cluster center
radii = distances.copy()
np.fill_diagonal(radii, np.inf)
radii = np.min(radii, axis=1) / 2
 
# Create a Plotly figure
fig_kdd = go.Figure()
 
 
 
 
 
# Add circles and center points for each cluster
for i, (center, radius) in enumerate(zip(cluster_centers, radii)):
    fig_kdd.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=center[0] - radius, y0=center[1] - radius,
                  x1=center[0] + radius, y1=center[1] + radius,
                  line_color="LightSeaGreen")
    fig_kdd.add_trace(go.Scatter(x=[center[0]], y=[center[1]], text=[f"{i}"],
                             mode="markers+text",
                             marker=dict(color="Black", size=12),
                             textposition="bottom center"))
 
# Update the layout for the figure
fig_kdd.update_layout(
    title="KMeans Intercluster Distance Map for KDD Cup 1999 Dataset",
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False)
)
 
# Append figure to the list
figures1.append(fig_kdd)
fig_kdd.show()
 
Kmeans.append(fig_kdd)
 
 
 
 
 
 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
 
figures = []
 
# Load the dataset
df = pd.read_csv('UNSW_NB15_testing-set.csv')
 
# Assuming 'proto', 'service', and 'state' are categorical features
label_encoder = LabelEncoder()
categorical_features = ['proto', 'service', 'state']
for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])
 
# Drop irrelevant features
drop_features = ['record_start_time', 'record_last_time', 'srcip', 'sport', 'dstip', 'dsport']
df.drop(columns=drop_features, inplace=True, errors='ignore')
 
# Define the target feature
target_feature = 'attack_cat'  # Update if different
 
# Separate features and target
X = df.drop(target_feature, axis=1)
y = df[target_feature]
 
# Perform stratified sampling to take 20% of the dataset
X_sampled, _, y_sampled, _ = train_test_split(X, y, test_size=0.9, stratify=y, random_state=42)
 
# Initialize Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
 
# Fit the model
rf.fit(X_sampled, y_sampled)
 
# Get feature importances and sort them
importances = rf.feature_importances_
features_df = pd.DataFrame({'Feature': X_sampled.columns, 'Importance': importances})
features_df = features_df.sort_values(by='Importance', ascending=False)
 
# Extracting top 24 features (or any number you prefer)
top_features_df = features_df.head(24)
 
# Extracting the names of the top features
top_features_names = top_features_df['Feature'].tolist()
 
# Creating a new dataset with only the top features and the target variable
top_features_dataset = df[top_features_names + [target_feature]]
 
# Displaying the first few rows of the new dataset with top features
print(top_features_dataset.head())
 
# Sampled dataset for visualization
sampled_dataset = top_features_dataset.copy()
 
 
fig_kddcup_pie = px.pie(
    sampled_dataset,
    names=sampled_dataset['attack_cat'],
    title='Distribution of Attack Types in UNSW-NB15 Dataset'
)
#fig_kddcup_pie.show()
figures.append(fig_kddcup_pie)
piecharts.append(fig_kddcup_pie)
 
 
# Calculate class frequencies and percentages
class_counts = sampled_dataset['attack_cat'].value_counts()
class_percentages = (class_counts / len(sampled_dataset)) * 100
 
# Create a DataFrame for Plotly Express
data_for_plot = pd.DataFrame({
    'Class': class_counts.index,
    'Frequency': class_counts.values,
    'Percentage': class_percentages.values
})
 
# Create the bar chart
fig1 = px.bar(data_for_plot, x='Class', y='Frequency', text='Percentage')
 
# Adjust the appearance
fig1.update_layout(
    title='Class Distribution of UNSW-NB15 Dataset',
    xaxis_title='Class',
    yaxis_title='Frequency',
    xaxis={'categoryorder': 'total descending'}
)
#fig1.show()
figures.append(fig1)
barplots.append(fig1)
 
 
# Corrected code to calculate and display Mahalanobis distance heatmaps using Plotly
 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
import plotly.express as px
# Load the dataset
dataset = pd.read_csv('UNSW_NB15_testing-set.csv')
 
# Remove redundant and irrelevant features
irrelevant_features = ['srcip', 'sport', 'dstip', 'dsport']
dataset.drop(columns=irrelevant_features, inplace=True, errors='ignore')
 
# Convert nominal features to numerical
nominal_features = ['proto', 'state', 'service']  # These features are categorical
label_encoders = {feature: LabelEncoder() for feature in nominal_features}
for feature in nominal_features:
    dataset[feature] = label_encoders[feature].fit_transform(dataset[feature])
 
# Define scalers
scalers = {
    "Original": None,
    "MinMax": MinMaxScaler(),
    "Quantile": QuantileTransformer(),
}
 
# Function to calculate Mahalanobis distance
def calculate_mahalanobis(data, labels):
    # Calculate centroids for each class
    centroids = data.groupby(labels).mean()
 
    # Compute covariance matrix
    cov_matrix = np.cov(data.T)
 
    # Regularize covariance matrix to avoid singularity
    regularization_factor = 1e-4
    regularized_cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * regularization_factor
 
    # Compute inverse covariance matrix
    inv_cov_matrix = np.linalg.inv(regularized_cov_matrix)
 
    # Calculate Mahalanobis distance between each pair of centroids
    distances = pd.DataFrame(index=centroids.index, columns=centroids.index)
    for i in centroids.index:
        for j in centroids.index:
            distances.at[i, j] = mahalanobis(centroids.loc[i], centroids.loc[j], inv_cov_matrix)
    return distances
 
 
 
 
# Calculate and plot Mahalanobis distance for each scaler
for name, scaler in scalers.items():
    if scaler:
        # Apply the scaler
        scaled_features = scaler.fit_transform(top_features_dataset.drop(['attack_cat', 'label'], axis=1))
    else:
        # If scaler is None, use the original features
        scaled_features = top_features_dataset.drop(['attack_cat', 'label'], axis=1).values
 
    # Get numeric features and class labels for Mahalanobis distance calculation
    numeric_features = pd.DataFrame(scaled_features, columns=top_features_dataset.drop(['attack_cat', 'label'], axis=1).columns)
    class_labels = top_features_dataset['attack_cat']
 
    # Calculate Mahalanobis distances
    m_distances = calculate_mahalanobis(numeric_features, class_labels)
 
    # Plot heatmap using Plotly
    fig = px.imshow(m_distances,
                    labels=dict(color="Mahalanobis Distance"),
                    x=m_distances.columns,
                    y=m_distances.index,
                    title=f"Mahalanobis Distance Heatmap - {name} for UNSW-NB15 Dataset")
    fig.update_xaxes(side="bottom")
    figures.append(fig)
    heatmaps.append(fig)
 
    #fig.show()
 
 
 
 
 
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
 
 
 
# Load the training and testing datasets
training_set_path = 'UNSW_NB15_training-set.csv'
testing_set_path = 'UNSW_NB15_testing-set.csv'
 
training_set = pd.read_csv(training_set_path)
testing_set = pd.read_csv(testing_set_path)
 
# Calculate the number of records for each category in the 'attack_cat' feature
training_counts = training_set['attack_cat'].value_counts()
testing_counts = testing_set['attack_cat'].value_counts()
 
# Summarizing the counts
total_counts = pd.concat([training_counts, testing_counts], axis=1, keys=['Training Subset', 'Testing Subset'])
total_counts.fillna(0, inplace=True)  # Replace NaN with 0
total_counts = total_counts.astype(int)  # Convert counts to integers
total_counts.loc['Total'] = total_counts.sum()  # Add a total count row
total_counts
 

 
 
 
class_counts = top_features_dataset['attack_cat'].value_counts()
class_percentages = (class_counts / len(top_features_dataset)) * 100
 
# Create a DataFrame for Plotly Express
data_for_plot = pd.DataFrame({
    'Class': class_counts.index,
    'Frequency': class_counts.values,
    'Percentage': class_percentages.values
})
 
# Create the bar chart
fig = px.bar(data_for_plot, x='Class', y='Frequency', text='Percentage')
 
# Adjust the appearance
fig.update_layout(
    title='Class Distribution of UNSW-NB15 Dataset',
    xaxis_title='Class',
    yaxis_title='Frequency',
    xaxis={'categoryorder':'total descending'}
)
 
 
# Show the plot
#fig.show()
 
# Separate features and labels for PCA
features = top_features_dataset.drop(['attack_cat', 'label'], axis=1)
labels = top_features_dataset['attack_cat']
 
# PCA to reduce to three dimensions
pca_3d = PCA(n_components=3)
features_pca_3d = pca_3d.fit_transform(features)
# Create a DataFrame for the PCA results
df_pca_3d = pd.DataFrame(features_pca_3d, columns=['PC1', 'PC2', 'PC3'])
df_pca_3d['attack_cat'] = labels.values
 
fig_3d = px.scatter_3d(df_pca_3d, x='PC1', y='PC2', z='PC3', color='attack_cat', title='3D PCA Visualization of UNSW-NB15 Dataset')
#fig_3d.show()
figures.append(fig_3d)
print("pca")
pca_plots.append(fig_3d)
 
 
 
features_for_parallel_plot = top_features_dataset.drop('attack_cat', axis=1)
 
# Create the parallel coordinates plot
fig_parallel = px.parallel_coordinates(
    features_for_parallel_plot,
    color=top_features_dataset['attack_cat'].astype('category').cat.codes,  # Encoding labels as colors
    labels={col: col for col in features_for_parallel_plot.columns},  # Customizing labels if needed
    title='Parallel Coordinates Plot for UNSW_NB15 dataset'
)
 
# Show the plot
#fig_parallel.show()
figures.append(fig_parallel)
parallel_coord.append(fig_parallel)
 
 
umap_3d = UMAP(n_components=3, init='random', random_state=0)
 
proj_3d = umap_3d.fit_transform(features)
 
# Convert the projections to DataFrames
df_proj_3d = pd.DataFrame(proj_3d, columns=['PC1', 'PC2', 'PC3'])
 
# Add the 'attack_cat' column (assuming 'attack_cat' is available in your context)
df_proj_3d['attack_cat'] = labels.values
 
# Plotting
 
 
fig_3d = px.scatter_3d(
    df_proj_3d, x='PC1', y='PC2', z='PC3',
    color='attack_cat',title = 'UMAP of UNSW-NB15 Dataset'
)
 
fig_3d.update_traces(marker_size=5)
figures.append(fig_3d)
print("umap")
 
umap.append(fig_3d)
 
 

# t-SNE for visualization
 
tsne = TSNE(n_components=2, perplexity=91, n_iter=1000, random_state=42, metric='euclidean')
features_tsne = tsne.fit_transform(features)

 
# Create a DataFrame for the t-SNE results
df_tsne = pd.DataFrame(features_tsne, columns=['Dimension 1', 'Dimension 2'])
df_tsne['attack_cat'] = labels.values
 
# Plotting t-SNE visualization for the entire dataset
fig_tsne = px.scatter(
    df_tsne,
    x='Dimension 1',
    y='Dimension 2',
    color='attack_cat',
    title='t-SNE Visualization of the UNSW-NB15 Dataset',
    labels={'color': 'Attack Category'}
)
figures.append(fig_tsne)
fig_tsne.show()
print("tsne")
tsne_list.append(fig_tsne)
 
 



 
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
 
# Perform PCA to reduce to two dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
 
# Run KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)  # Adjust the number of clusters if necessary
kmeans.fit(X_pca)
cluster_centers = kmeans.cluster_centers_
 
# Calculate the pairwise distances between cluster centers
distances = pairwise_distances(cluster_centers)
 
# Determine the radius of the circles to draw around each cluster center
# The radius is half of the distance to the nearest cluster center
radii = distances.copy()
np.fill_diagonal(radii, np.inf)
radii = np.min(radii, axis=1) / 2
 
# Create a Plotly figure for the KMeans Intercluster Distance Map
fig = go.Figure()
 
# Add circles and center points for each cluster
for i, (center, radius) in enumerate(zip(cluster_centers, radii)):
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=center[0] - radius, y0=center[1] - radius,
                  x1=center[0] + radius, y1=center[1] + radius,
                  line_color="LightSeaGreen",
                 )
    fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], text=[f"{i}"],
                             mode="markers+text",
                             marker=dict(color="Black", size=12),
                             textposition="bottom center"))
 
# Update the layout for the KMeans Intercluster Distance Map
fig.update_layout(
    title="KMeans Intercluster Distance Map for UNSW-NB15 Dataset",
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False)
)
 
# Show the KMeans Intercluster Distance Map
fig.show()
figures.append(fig)
Kmeans.append(fig)
 

import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
 
 
# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
 
app.layout = html.Div([
    dbc.NavbarSimple(brand="My Data Visualization App", color="primary", dark=True),
    html.Br(),
    dbc.Row([
        dbc.Col(dbc.Button("Piecharts", id="btn-piecharts", n_clicks=0, color="info"), width=2),
        dbc.Col(dbc.Button("Distribution Plots", id="btn-barplots", n_clicks=0, color="warning"), width=2),
        dbc.Col(dbc.Button("Mahalanobis Plots", id="btn-heatmaps", n_clicks=0, color="success"), width=2),
        dbc.Col(dbc.Button("PCA Plots", id="btn-pca", n_clicks=0, color="danger"), width=2),
        dbc.Col(dbc.Button("T-sne Plots", id="btn-tsne", n_clicks=0, color="dark"), width=2),
        dbc.Col(dbc.Button("Umap Plots", id="btn-umap", n_clicks=0, color="warning"), width=2),
        dbc.Col(dbc.Button("Parallel Coordinate Plots", id="btn-pcod", n_clicks=0, color="dark"), width=2),
        dbc.Col(dbc.Button("Kmeans Plots", id="btn-kmean", n_clicks=0, color="warning"), width=2),
    ], justify="center"),  # This will center the row content
    html.Br(),  # Space before the next row
    dbc.Row([
        dbc.Col(dbc.Button("Dashboard for KDD Cup 1999 Dataset", id="btn-combined1", n_clicks=0, color="secondary"), width={"size": 3, "offset": 3}),
        dbc.Col(dbc.Button("Dashboard for UNSW-NB15 Dataset", id="btn-combined2", n_clicks=0, color="dark"), width=3),
    ], justify="center"),  # This will center the row content
    html.Hr(),
    html.Div(id="plot-container", style={'padding': '20px'})
])
 
  
# Callback to display plots
@app.callback(
    Output("plot-container", "children"),
    [Input("btn-barplots", "n_clicks"), Input("btn-heatmaps", "n_clicks"),
     Input("btn-pca", "n_clicks"),  Input("btn-piecharts", "n_clicks"),
     Input("btn-umap","n_clicks"),  Input("btn-tsne","n_clicks"),
     Input("btn-pcod","n_clicks"),Input("btn-kmean","n_clicks"),
     Input("btn-combined1", "n_clicks"), Input("btn-combined2", "n_clicks")]
)
def display_plots(btn_bar, btn_heatmap, btn_piecharts,btn_pca,btn_tsne, btn_umap,btn_pcod,btn_kmean,btn_combined1, btn_combined2):
    ctx = dash.callback_context
 
    if not ctx.triggered:
        return "Click a button to display plots."
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
 
    if button_id == "btn-barplots":
        return [dcc.Graph(figure=fig) for fig in barplots]
    elif button_id == "btn-piecharts":
        return [dcc.Graph(figure=fig) for fig in piecharts]
    elif button_id == "btn-heatmaps":
        heatmaps_per_row = 3
        rows = [heatmaps[i:i + heatmaps_per_row] for i in range(0, len(heatmaps), heatmaps_per_row)]
 
        heatmap_rows = []
        for row in rows:
            heatmap_rows.append(dbc.Row([dbc.Col(dcc.Graph(figure=fig), md=6) for fig in row]))
 
        return heatmap_rows
    elif button_id == "btn-umap":
        return [dcc.Graph(figure=fig) for fig in umap]
    elif button_id == "btn-tsne":
        return [dcc.Graph(figure=fig) for fig in tsne_list]
        #return [dcc.Graph(figure=fig) for fig in heatmaps]
    elif button_id == "btn-pca":
        return [dcc.Graph(figure=fig) for fig in pca_plots]
    elif button_id == "btn-pcod":
        return [dcc.Graph(figure=fig) for fig in parallel_coord]
    elif button_id == "btn-kmean":
        return [dcc.Graph(figure=fig) for fig in Kmeans]
    elif button_id == "btn-combined1":
        return dbc.Row([dbc.Col(dcc.Graph(figure=fig), md=6) for fig in figures1])
        #return [dcc.Graph(figure=fig) for fig in figures1]
    elif button_id == "btn-combined2":
        return dbc.Row([dbc.Col(dcc.Graph(figure=fig), md=6) for fig in figures])
 
        #return [dcc.Graph(figure=fig) for fig in figures]
 
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)