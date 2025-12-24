import pandas as pd

# Load the four CSV files
df1 = pd.read_csv('News_Domain_Classification/dataset/file1.csv')  # HuffPost (text, category)
df2 = pd.read_csv('News_Domain_Classification/dataset/file2.csv', sep=';')  # Newscatcher
df3 = pd.read_csv('News_Domain_Classification/dataset/file3.csv')  # MN-DS style
df4 = pd.read_csv('News_Domain_Classification/dataset/file4.csv')  # AG News

# Process DF1: HuffPost 
df1['text'] = df1['text'].fillna('')
df1['category'] = df1['category'].str.upper().str.strip()

# Process DF2: Newscatcher 
df2['text'] = df2['title'].fillna('') + ' ' + df2.get('link', pd.Series([''] * len(df2))).fillna('')
df2['category'] = df2['topic'].str.upper().str.strip()

# Process DF3: MN-DS style (title + content) 
df3['text'] = df3['title'].fillna('') + ' ' + df3['content'].fillna('')
df3['category'] = df3['category_level_1'].str.upper().str.strip()

# Process DF4: AG News (Class Index, Title, Description) 
# Clean column names (remove any hidden spaces)
df4.columns = df4.columns.str.strip()

# Map Class Index to your categories
class_map = {
    1: 'WORLD NEWS',
    2: 'SPORTS',
    3: 'BUSINESS',
    4: 'TECH'   # Sci/Tech → TECH (or SCIENCE if you prefer; most are tech)
}
df4['category'] = df4['Class Index'].map(class_map)
df4['text'] = df4['Title'].fillna('') + ' ' + df4['Description'].fillna('')
df4 = df4[['text', 'category']]  # Keep only these
df4['category'] = df4['category'].str.upper()

# Global category cleanup/mapping 
category_mapping = {
    'SCI/TECH': 'TECH',
    'SCIENCE AND TECHNOLOGY': 'SCIENCE',
    'TECHNOLOGY': 'TECH',
    'WORLD': 'WORLD NEWS',
    'HEALTH': 'HEALTHY LIVING',
    'WELLNESS': 'HEALTHY LIVING',
    'FOOD AND DRINK': 'FOOD & DRINK',
    'THE WORLDPOST': 'WORLD NEWS',
    'WORLDPOST': 'WORLD NEWS',
    'NATION': 'WORLD NEWS',
    'CRIME, LAW AND JUSTICE': 'POLITICS',
    'CONFLICT, WAR AND PEACE': 'WORLD NEWS',
    'ARTS, CULTURE, ENTERTAINMENT AND MEDIA': 'ENTERTAINMENT',
    'SPORT': 'SPORTS',
}

for df in [df1, df2, df3, df4]:
    df['category'] = df['category'].replace(category_mapping)

# 10 target categories ===
target_categories = [
    'BUSINESS', 'ENTERTAINMENT', 'POLITICS', 'SPORTS', 'TECH',
    'SCIENCE', 'TRAVEL', 'FOOD & DRINK', 'HEALTHY LIVING', 'WORLD NEWS'
]

# Filter only rows with valid categories
df1 = df1[df1['category'].isin(target_categories)]
df2 = df2[df2['category'].isin(target_categories)]
df3 = df3[df3['category'].isin(target_categories)]
df4 = df4[df4['category'].isin(target_categories)]

# Merge all 
df_merged = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Clean: remove empty text and duplicates
df_merged['text'] = df_merged['text'].str.strip()
df_merged = df_merged[df_merged['text'] != '']
df_merged = df_merged.drop_duplicates(subset=['text'])

# Final dataset
df_final = df_merged[['text', 'category']]

# Save
df_final.to_csv('News_Domain_Classification/dataset/domain_classification_dataset_merged.csv', index=False)

# Summary
print("Merge successful!")
print(f"Total samples: {len(df_final)}")
print("\nCategory distribution:")
print(df_final['category'].value_counts().sort_values(ascending=False))