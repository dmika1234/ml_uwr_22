

### One-hot encode in one-go
```python
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the vectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=100)

# fit_transform the vectorizer to your numpy array column
raw_data.fillna('', inplace = True)
X = vectorizer.fit_transform(raw_data['company_profile'])
df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names_out())
```