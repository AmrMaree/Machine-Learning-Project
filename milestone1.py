# # **Importing libraries**

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# -----------

# # **Data Inspecting**

info_df = pd.read_csv('info_base_games.csv')
gamalytic_df = pd.read_csv('gamalytic_steam_games.csv')
dlcs_df = pd.read_csv('dlcs.csv')
demos_df = pd.read_csv('demos.csv')

info_df.head()

info_df.info()

info_df['release_date'].value_counts().head(30)

gamalytic_df.head()

gamalytic_df.info()

dlcs_df.head()

dlcs_df.info()

demos_df.head()

demos_df.info()

# ----------------

# # **Data Wrangling**

# ### *Renaming columns to have a common 'id' column for merging*

info_df.rename(columns={'appid': 'id'}, inplace=True)
gamalytic_df.rename(columns={'steamId': 'id'}, inplace=True)
dlcs_df.rename(columns={'base_appid': 'id'}, inplace=True)
demos_df.rename(columns={'full_game_appid': 'id'}, inplace=True)

# ### *Ensure consistent ID types*


info_df['id'] = info_df['id'].astype(str)
gamalytic_df['id'] = gamalytic_df['id'].astype(str)
dlcs_df['id'] = dlcs_df['id'].astype(str)
demos_df['id'] = demos_df['id'].astype(str)

# ### **Data Merging**

# Merge info_df and gamalytic_df
merged_df = pd.merge(info_df, gamalytic_df, on='id', how='inner')

# Aggregate DLCs (count per game)
dlc_count = dlcs_df.groupby('id').size().reset_index(name='dlc_count')

merged_df = pd.merge(merged_df, dlc_count, on='id', how='left')
merged_df['dlc_count'] = merged_df['dlc_count'].fillna(0)

# Add demo presence
merged_df['hasDemo'] = merged_df['id'].isin(demos_df['id']).astype(int)

merged_df.info()

merged_df.isnull().sum()

# merged_df.to_csv('final_merged_data.csv', index=False)

df = merged_df.copy()

# ### *Handle missing values*

df['metacritic'] = pd.to_numeric(df['metacritic'], errors='coerce').fillna(0)
df['achievements_total'] = pd.to_numeric(df['achievements_total'], errors='coerce').fillna(0)
df['genres'] = df['genres'].fillna('Unknown')
df['release_date'] = df['release_date'].replace('Coming soon', pd.NA)
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year.fillna(df['release_date'].dt.year.mode()[0])

# Drop aiContent if all NaN
df.drop(columns=['aiContent'], inplace=True)

df.isnull().sum()

df['release_year'].value_counts()

df['publisherClass'].value_counts()

# # **Data Preprocessing**

# ## ***Encoding & Feature Engineering***

# Encode categorical variables
df['steam_achievements'] = df['steam_achievements'].astype(int)
df['steam_trading_cards'] = df['steam_trading_cards'].astype(int)
df['workshop_support'] = df['workshop_support'].astype(int)
df = pd.get_dummies(df, columns=['publisherClass'], dtype=int)

# Parse supported_platforms
df['isWindows'] = df['supported_platforms'].apply(lambda x: 1 if 'windows' in str(x).lower() else 0)
df['isMac'] = df['supported_platforms'].apply(lambda x: 1 if 'mac' in str(x).lower() else 0)
df['isLinux'] = df['supported_platforms'].apply(lambda x: 1 if 'linux' in str(x).lower() else 0)

df.info()

# --------------

# # **Exploratory Data Analysis**

# ## ***Uni-Variate Analysis***

# ### MetaCritic Column

plt.figure(figsize=(8, 6))
plt.style.use('ggplot')
sns.histplot(data=df, x='metacritic', bins=10, kde=True, color='blue')
plt.title('Distribution of Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# ### Steam Achievements Column

df['steam_achievements'].value_counts()

sns.countplot(data=df, x='steam_achievements', palette='viridis') 
plt.title('Count of Each Category')
plt.xlabel('Steam Achievements')
plt.ylabel('Count')
plt.show()

# ### Steam Trading Cards Column

df['steam_trading_cards'].value_counts()

sns.countplot(data=df, x='steam_trading_cards', palette='viridis') 
plt.title('Count of Each Category')
plt.xlabel('Steam Trading Cards')
plt.ylabel('Count')
plt.show()

# ### Workshop Support Column

df['workshop_support'].value_counts()

sns.countplot(data=df, x='workshop_support', palette='viridis') 
plt.title('Count of Each Category')
plt.xlabel('Workshop Support')
plt.ylabel('Count')
plt.show()

# ### Genres Column

df['genres'].value_counts()

top_10_genre=df['genres'].value_counts().head(10)

plt.figure(figsize=(12,6))
sns.barplot(x=top_10_genre.index, y=top_10_genre.values, palette='viridis')
plt.title('Top 10 Genres')
plt.xlabel('Genre Name')
for i, count in enumerate(top_10_genre.values):
    plt.text(i, count + 15, str(count), ha='center')
plt.xticks(rotation=45)
plt.show()

# ### Achievements Total Column

top10ach=df['achievements_total'].value_counts().head(10)

plt.figure(figsize=(12,6))
plt.style.use('ggplot')
sns.barplot(x=top10ach.index, y=top10ach.values, palette='viridis')
plt.title('Top numbers of Achievements')
plt.xlabel('Achiements Total')
for i, count in enumerate(top10ach.values):
    plt.text(i, count + 15, str(count), ha='center')
plt.xticks(rotation=45)
plt.show()

# ### Release Date Column

topDate=df['release_date'].value_counts().head(10)

plt.figure(figsize=(12,6))
plt.style.use('ggplot')
sns.barplot(x=topDate.index, y=topDate.values, palette='viridis')
plt.title('Top 10 Dates')
plt.xlabel('Release Date')
for i, count in enumerate(topDate.values):
    plt.text(i, count + 2, str(count), ha='center')
plt.xticks(rotation=45)
plt.show()

# ### Price Column

# - $50–$70	Normal	New AAA games
# - $100	Normal (for bundles or deluxe editions)	
# - $150+	Rare (big bundles, collector's editions, or mistakes)	
# - $500–$1900	Not normal (real outliers)

df['price'].describe()

# Calculate Q1, Q3, and IQR
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier thresholds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print(lower_bound)
print(upper_bound)

# Remove outliers
#df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

df[df['price']>199.99].T

plt.figure(figsize=(8,5))
plt.boxplot(df.price, vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))
plt.title('Boxplot of Price')
plt.xlabel('Price')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# - Most Steam games are between $0.99 – $19.99 normally.

# - Games priced $29.99, $49.99, $59.99 (AAA games) are rare compared to indie games.

df[[df['publisherClass_Hobbyist'] == 1] and df['price'] > 200]

print((df['price'] == 0).sum())

print((df['price'] > 0).sum())

print((df['price'] >50 ).sum())

print((df['price'] > 70  ).sum())


df = df[df['price'] <= 70]

plt.figure(figsize=(8,5))
plt.boxplot(df.price, vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))
plt.title('Boxplot of Price')
plt.xlabel('Price')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

df.info()

# - most of the games are free

# ### Copies Sold Column

top10copies=df['copiesSold'].value_counts().head(10)

top10copies.index = top10copies.index.astype(str)

plt.figure(figsize=(12,6))
plt.style.use('ggplot')
sns.barplot(x=top10copies.index, y=top10copies.values, palette='viridis')
plt.title('Top 10 Copies Sold Total')
plt.xlabel('Items')   
plt.ylabel('Copies Sold Total')
plt.xticks(rotation=45)
for i, count in enumerate(top10copies.values):
    plt.text(i, count + 50, str(count), ha='center')

plt.show()

# ### Review Score Column

top10rev=df['reviewScore'].value_counts().head(10)
top10rev.index = top10rev.index.astype(str)

#Display the top 10 review scores
plt.figure(figsize=(12,6))
plt.style.use('ggplot')
sns.barplot(x=top10rev.index, y=top10rev.values, palette='viridis')
plt.title('Top 10 Review Scores')
plt.xlabel('Review Score')
for i, count in enumerate(top10rev.values):
    plt.text(i, count + 15, str(count), ha='center')
plt.show()

# ### DLC Count Column

df['dlc_count'].value_counts()

sns.countplot(data=df, x='dlc_count', palette='viridis') 
plt.title('Count of Each Category')
plt.xlabel('DLC Count')
plt.ylabel('Count')
plt.show()


# ### Has Demo Column

df['hasDemo'].value_counts()    

sns.countplot(data=df, x='hasDemo', palette='viridis') 
plt.title('Count of Each Category')
plt.xlabel('Has Demo')
plt.ylabel('Count')
plt.show()


# ### Release Year Column

top10year=df['release_year'].value_counts()
top10year.index = top10year.index.astype(str)

#Display the top 10 review scores
plt.figure(figsize=(12,6))
plt.style.use('ggplot')
sns.barplot(x=top10year.index, y=top10year.values, palette='viridis')
plt.title('Top 10 Release Years')
plt.xlabel('Release Year')
for i, count in enumerate(top10year.values):
    plt.text(i, count + 15, str(count), ha='center')

plt.xticks(rotation=45)
plt.show()

# - Most of the games New

# ### Publisher Classes Columns

publisher_counts = {
    'AA': df['publisherClass_AA'].sum(),
    'AAA': df['publisherClass_AAA'].sum(),
    'Hobbyist': df['publisherClass_Hobbyist'].sum(),
    'Indie': df['publisherClass_Indie'].sum()
}
# Plot
plt.figure(figsize=(8, 5))
plt.bar(publisher_counts.keys(), publisher_counts.values(), color='skyblue')
plt.title("Number of Games by Publisher Class")
plt.xlabel("Publisher Class")
plt.ylabel("Number of Games")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ### Platforms Columns

platform_counts = {
    'Windows': df['isWindows'].sum(),
    'Mac': df['isMac'].sum(),
    'Linux': df['isLinux'].sum()
}
# Plot
plt.figure(figsize=(8, 5))
plt.bar(platform_counts.keys(), platform_counts.values(), color='skyblue')
plt.title("Number of Games Supporting Each Platform")
plt.xlabel("Platform")
plt.ylabel("Number of Games")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ## **Bi-Variate Analysis**

# ### ***1. Group by name***

# - combine name	Sum or Average of copiesSold	Find Top Selling Games

grouped = df.groupby('name')['copiesSold'].sum()
top10copies = grouped.sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
plt.style.use('ggplot')
sns.barplot(x=top10copies.index, y=top10copies.values, palette='viridis')
plt.title('Top 10 Games by Copies Sold')
plt.xlabel('Game Name')
plt.ylabel('Copies Sold Total')
plt.xticks(rotation=45)
for i, count in enumerate(top10copies.values):
    plt.text(i, count + 10000000, str(count), ha='center')
plt.show()

df.groupby('name').agg({
    'copiesSold': 'sum',
    'reviewScore': 'mean'
}).sort_values('copiesSold', ascending=False).head()

# ### ***2. Group by release_year***

grouped = df.groupby('release_year').agg({
    'copiesSold': 'sum',
    'reviewScore': 'mean',
    'id': 'count'  # Number of games released that year
}).sort_values('copiesSold', ascending=False)
grouped.index = grouped.index.astype(str)

# ### Release Year Vs Copies sold

plt.figure(figsize=(14, 6))
sns.barplot(x=grouped.index, y=grouped['copiesSold'], palette="viridis")
plt.title('Total Copies Sold per Release Year', fontsize=16)
plt.xlabel('Release Year')
plt.ylabel('Copies Sold')
plt.xticks(rotation=45)
plt.show()

# ### Release Year Vs Score

plt.figure(figsize=(14, 6))
sns.lineplot(x=grouped.index, y=grouped['reviewScore'], marker='o', color='green')
plt.title('Average Review Score per Release Year', fontsize=16)
plt.xlabel('Release Year')
plt.ylabel('Average Review Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# ### ***3. Group by genres***

grouped = df.groupby('genres').agg({
    'copiesSold': 'sum',
    'price': 'mean',
    'reviewScore': 'mean'
}).sort_values('copiesSold', ascending=False)

# ### Genres Vs Copies Sold

top5 = grouped.head()
plt.figure(figsize=(10, 4))
sns.barplot(y=top5.index, x=top5['copiesSold'], palette="viridis")
plt.title('Most sold game genres', fontsize=16)
plt.ylabel('Genres')
plt.xlabel('Copies Sold')
plt.xticks(rotation=45)
plt.show()

# ### Genres vs Price

top5 = grouped.head()
plt.figure(figsize=(10, 4))
sns.barplot(x=top5.index, y=top5['price'], palette="viridis")
plt.title('Genres Vs Price', fontsize=16)
plt.ylabel('price')
plt.xlabel('genres')
plt.xticks(rotation=45)
plt.show()

# ### **Scaling**

scaler = StandardScaler()
df['copiesSold'] = scaler.fit_transform(df[['copiesSold']])

# ---------------------------

df.info()

df.head().T

# # **Feature Selection**

features = [
    'metacritic',
    'steam_achievements',
    'steam_trading_cards',
    'workshop_support',
    'achievements_total',
    'price',
    'dlc_count',
    'copiesSold',
    'hasDemo',
    'release_year',
    'publisherClass_AA',
    'publisherClass_AAA',
    'publisherClass_Hobbyist',
    'publisherClass_Indie',
    'isWindows',
    'isMac',
    'isLinux'
]

X = df[features]
y = df['reviewScore']

# # **Data Spliting**

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=57,shuffle=True)

# # **Modeling**

models = {
    'XGBoost Regressor': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100),
    'LightGBM Regressor': lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1,verbose = 0)
}

results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)  
    test_score = model.score(X_test, y_test)  
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    results.append({
        'Model': model_name,
        'Train Score': train_score,
        'Test Score': test_score,
        'Mean Squared Error': mse
    })
results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  
plt.xlabel('Actual Review Scores')
plt.ylabel('Predicted Review Scores')
plt.title('Actual vs Predicted')
plt.grid(True)
plt.show()

# ### Tried to use web Scrapping

# import requests
# from bs4 import BeautifulSoup
# import re
# import json

# def scrape_steam_game(appid):
#     url = f"https://store.steampowered.com/app/{appid}/"
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124",
#         "Cookie": "birthtime=0; lastagecheckage=1-January-1970"
#     }
    
#     try:
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
#     except requests.RequestException as e:
#         return {"error": f"Failed to fetch page: {str(e)}"}
    
#     soup = BeautifulSoup(response.text, "html.parser")
#     result = {}
    
#     # Name
#     result["name"] = soup.find("div", class_="apphub_AppName").get_text(strip=True) if soup.find("div", class_="apphub_AppName") else None
    
#     # Metacritic
#     metacritic = soup.find("div", class_=re.compile(r"score (high|medium|low)"))
#     result["metacritic"] = int(metacritic.get_text(strip=True)) if metacritic else None
    
#     # Steam Achievements
#     result["steam_achievements"] = bool(soup.find("div", class_="game_area_achievement_stats"))
    
#     # Steam Trading Cards
#     result["steam_trading_cards"] = bool(soup.find("a", href=re.compile(r"tradingcards")))
    
#     # Workshop Support
#     result["workshop_support"] = bool(soup.find("a", href=re.compile(r"steamcommunity\.com\/app\/\d+\/workshop")))
    
#     # Genres
#     genres = [a.get_text(strip=True) for a in soup.find_all("a", class_="app_tag") if a.get_text(strip=True).lower() != "indie"]
#     result["genres"] = genres if genres else None
    
#     # Achievements Total
#     achievements = soup.find("span", class_="communitylink_achievement_count")
#     result["achievements_total"] = int(achievements.get_text(strip=True)) if achievements else None
    
#     # Release Date
#     release_date = soup.find("div", class_="release_date")
#     result["release_date"] = release_date.find("div", class_="date").get_text(strip=True) if release_date else None
    
#     # Supported Platforms
#     platforms = []
#     if soup.find("div", class_="sysreq_tab", attrs={"data-os": "win"}): platforms.append("Windows")
#     if soup.find("div", class_="sysreq_tab", attrs={"data-os": "mac"}): platforms.append("Mac")
#     if soup.find("div", class_="sysreq_tab", attrs={"data-os": "linux"}): platforms.append("Linux")
#     result["supported_platforms"] = platforms if platforms else None
    
#     # Price
#     price = soup.find("div", class_="game_purchase_price")
#     result["price"] = price.get_text(strip=True) if price else "Free"
    
#     # Copies Sold
#     result["copiesSold"] = None  # Not available on store page
    
#     # Publisher Class
#     publisher = soup.find("div", id="developers_list")
#     if publisher:
#         publisher_name = publisher.find("a").get_text(strip=True)
#         aaa_publishers = ["Valve", "Bethesda", "Rockstar", "EA", "Ubisoft", "Activision"]
#         result["publisherClass"] = "AAA" if publisher_name in aaa_publishers else "Indie"
#     else:
#         result["publisherClass"] = None
    
#     # Review Score
#     review_summary = soup.find("span", class_="game_review_summary")
#     review_percent = soup.find("span", class_="nonresponsive_hidden responsive_reviewdesc")
#     review_score = review_summary.get_text(strip=True) if review_summary else None
#     if review_percent:
#         percent = re.search(r"\d+%", review_percent.get_text(strip=True))
#         review_score += f" ({percent.group()})" if percent else ""
#     result["reviewScore"] = review_score
    
#     # AI Content
#     description = soup.find("div", class_="game_area_description")
#     result["aiContent"] = bool(description and ("ai-generated" in description.get_text(strip=True).lower() or "artificial intelligence" in description.get_text(strip=True).lower()))
    
#     # DLC App IDs
#     dlc_section = soup.find("div", class_="gameDlcBlocks")
#     dlc_appids = [a["href"].split("/")[4] for a in dlc_section.find_all("a", href=re.compile(r"store\.steampowered\.com\/app\/\d+"))] if dlc_section else None
#     result["dlc_appid"] = dlc_appids
    
#     # Demo App ID
#     demo = soup.find("a", href=re.compile(r"store\.steampowered\.com\/app\/\d+\/.*demo"))
#     result["demo_appid"] = demo["href"].split("/")[4] if demo else None
    
#     return result

# # Example usage
# if __name__ == "__main__":
#     appid = "730"  # Counter-Strike 2
#     data = scrape_steam_game(appid)
#     print(json.dumps(data, indent=2))