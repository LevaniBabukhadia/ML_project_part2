from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklear.metrics import precision_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
svm_model = SVC().fit(X_train, y_train)
lr_model = LogisticRegression().fit(X_train, y_train)
knn_model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
kmeans_model = KMeans(n_clusters=3).fit(X_train)

