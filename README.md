# akin

Some text similarity utilities

```python
def sort_dataframe(dataf, featurizer, examples, text_col="text", dist_col="dist", metric="euclidean"):
    X_tfm = featurizer.fit_transform(examples)
    X_other = tfm.transform(list(dataf[text_col]))
    distances = pairwise_distances(X_tfm, X_other, metric=metric).mean(axis=0)
    return dataf.assign(**{dist_col: distances}).sort_values("dist")
```


```python
import numpy as np

from functools import reduce
from operator import concat
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import make_union
from sklearn.feature_extraction.text import CountVectorizer


class AkinClassifier:
    def __init__(self, examples, featurizer=None, mutually_exclusive=False, distance="euclidean", reducer="mean"):
        self.featurizer = featurizer
        if not featurizer:
            self.featurizer = make_union(
                CountVectorizer(), 
                CountVectorizer(analyzer="char", ngram_range=(2, 3))
            )
        texts = reduce(lambda x,y: concat(x, y), examples.values())
        self.featurizer.fit(texts)
        self.examples = examples
        self.reducer = reducer
        self.examples_tfm = {k: self.featurizer.transform(v) for k, v in examples.items()}

    def pipe(self, stream):
        for item in stream:
            yield {"text": item, **self.predict_single(item)}

    def predict_single(self, text):
        X_tfm = self.featurizer.transform([text])
        reducer_func = np.mean if self.reducer == "mean" else np.min
        return {k: reducer_func(pairwise_distances(X_tfm, self.examples_tfm[k])) for k in self.examples}
        
examples = {"positive": ["thanks so much", "compliment", "i like this!"], "negative": ["this stinks", "you suck"]}
clf = AkinClassifier(examples, featurizer=tfm)
```

```python
import pandas as pd 

path = "/home/vincent/Development/customer-support/tesco_support.csv"
df = pd.read_csv(path)

pd.set_option('display.max_colwidth', None)

texts = list(df['text'].head(1000))
pd.DataFrame(clf.pipe(texts)).sort_values("positive").head(20)
```
