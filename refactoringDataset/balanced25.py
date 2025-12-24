import pandas as pd
import nltk
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

#Step 1: Download NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

#Step 2: Synonym Replacement Function 
def synonym_replacement(text, n_replace_ratio=0.2):
    """Replace ~20% of eligible words with synonyms"""
    words = word_tokenize(text.lower())  # Lowercase for consistency
    tagged = pos_tag(words)
    
    new_words = words.copy()
    eligible = []  # Indices of nouns, verbs, adjectives, adverbs
    
    for i, (word, tag) in enumerate(tagged):
        if tag.startswith(('NN', 'VB', 'JJ', 'RB')) and word.isalpha():
            eligible.append(i)
    
    if not eligible:
        return text 
    
    num_replace = max(1, int(len(eligible) * n_replace_ratio))
    replace_idx = random.sample(eligible, min(num_replace, len(eligible)))
    
    for i in replace_idx:
        word = words[i]
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != word.lower() and ' ' not in name:
                    synonyms.add(name)
        
        if synonyms:
            new_words[i] = random.choice(list(synonyms))
    
    return ' '.join(new_words).capitalize()

#Step 3: Load your unbalanced merged dataset 
df = pd.read_csv('News_Domain_Classification/dataset/domain_classification_dataset_merged.csv')

# Clean text
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'] != '']

print(f"Original total samples: {len(df)}")
print("Original distribution:")
print(df['category'].value_counts())

#Step 4: Define targets  
target_per_category = 25000
target_categories = [
    'BUSINESS', 'ENTERTAINMENT', 'POLITICS', 'SPORTS', 'TECH',
    'SCIENCE', 'TRAVEL', 'FOOD & DRINK', 'HEALTHY LIVING', 'WORLD NEWS'
]

# Minority categories that need augmentation
minority_categories = ['SCIENCE', 'FOOD & DRINK', 'TRAVEL', 'HEALTHY LIVING', 'ENTERTAINMENT']

augmented_parts = [df] 

print("\nStarting augmentation for minority categories...")

for cat in minority_categories:
    df_cat = df[df['category'] == cat].copy()
    current = len(df_cat)
    
    if current >= target_per_category:
        print(f"{cat}: {current} (already enough, no augmentation needed)")
        continue
    
    needed = target_per_category - current
    print(f"{cat}: {current} → generating {needed} new augmented samples")
    
    # Sample existing rows (with replacement) and augment them
    base_samples = df_cat.sample(n=needed, replace=True, random_state=42)
    
    new_texts = []
    for text in base_samples['text']:
        augmented_text = synonym_replacement(text)
        new_texts.append(augmented_text)
    
    # Create new augmented rows
    new_rows = pd.DataFrame({
        'text': new_texts,
        'category': [cat] * needed
    })
    
    augmented_parts.append(new_rows)

# Combine original + all augmented
df_augmented = pd.concat(augmented_parts, ignore_index=True)

print(f"\nAfter augmentation: {len(df_augmented)} samples")

#Step 5: Undersample all categories to exactly 25,000  
balanced_parts = []
for cat in target_categories:
    df_cat = df_augmented[df_augmented['category'] == cat]
    df_sampled = df_cat.sample(n=target_per_category, random_state=42)
    balanced_parts.append(df_sampled)

df_final = pd.concat(balanced_parts, ignore_index=True)

# Shuffle final dataset
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

#Step 6: Save  
output_path = 'News_Domain_Classification/dataset/domain_classification_dataset.csv'
df_final.to_csv(output_path, index=False)

#Step 7: Final Summary  
print("\n" + "="*50)
print("BALANCED DATASET CREATED SUCCESSFULLY!")
print("="*50)
print(f"Total samples: {len(df_final):,}")
print(f"Samples per category: {target_per_category:,}")
print("\nFinal distribution:")
print(df_final['category'].value_counts().sort_index())
print(f"\nSaved to: {output_path}")