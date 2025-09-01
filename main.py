import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_movies = 100
data = {
    'RottenTomatoes_Critics': np.random.randint(30, 100, num_movies),
    'RottenTomatoes_Audience': np.random.randint(20, 100, num_movies),
    'SocialMediaSentiment': np.random.normal(0, 1, num_movies), # 0 = neutral
    'BoxOfficeRevenue': 1000000 + 50000 * np.random.normal(0, 1, num_movies) + 10000 * (np.random.randint(0, 100, num_movies)) + 5000 * np.random.normal(0, 1, num_movies)
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
# Handle potential outliers (replace with a more robust method if needed)
df['BoxOfficeRevenue'] = np.clip(df['BoxOfficeRevenue'], 0, 10000000) #Cap at 10M
# --- 3. Feature Engineering ---
# No additional features needed for this simplified example.
# --- 4. Model Training ---
X = df[['RottenTomatoes_Critics', 'RottenTomatoes_Audience', 'SocialMediaSentiment']]
y = df['BoxOfficeRevenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# --- 5. Model Evaluation (Simplified) ---
# In a real-world scenario, use more comprehensive metrics.
r_sq = model.score(X_train, y_train)
print(f"R-squared (Training): {r_sq}")
# --- 6. Visualization ---
plt.figure(figsize=(10, 6))
sns.regplot(x=df['RottenTomatoes_Critics'], y=df['BoxOfficeRevenue'])
plt.title('Box Office Revenue vs. Critics Score')
plt.xlabel('Rotten Tomatoes Critics Score')
plt.ylabel('Box Office Revenue')
plt.savefig('critic_revenue.png')
print("Plot saved to critic_revenue.png")
plt.figure(figsize=(10, 6))
sns.regplot(x=df['SocialMediaSentiment'], y=df['BoxOfficeRevenue'])
plt.title('Box Office Revenue vs. Social Media Sentiment')
plt.xlabel('Social Media Sentiment')
plt.ylabel('Box Office Revenue')
plt.savefig('sentiment_revenue.png')
print("Plot saved to sentiment_revenue.png")
# --- 7. Prediction (Example) ---
# Predict box office revenue for a new movie
new_movie = pd.DataFrame({
    'RottenTomatoes_Critics': [85],
    'RottenTomatoes_Audience': [78],
    'SocialMediaSentiment': [0.8]
})
predicted_revenue = model.predict(new_movie)
print(f"Predicted Box Office Revenue for new movie: ${predicted_revenue[0]:,.2f}")