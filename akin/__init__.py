import numpy as np
import pandas as pd

from functools import reduce
from operator import concat
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import make_union
from sklearn.feature_extraction.text import CountVectorizer


class AkinClassifier:
    """
    Zero shot classifier for text data that uses a few examples to guess a class. 

    Arguments:
    - `examples`: dictionary of text examples, per label, to compare against
    - `featurizer`: a sklearn compatible featurizer, if none is given we use a set of counvectorizers
    - `metric`: the metric to use for distance calculation, default is "euclidean"
    - `reducer`: the method to use to reduce the distances of all examples to single text, default is "min"

    Usage:

    ```python
    import pandas as pd 
    from akin import AkinClassifier

    examples = {
        "positive": ["thanks so much", "compliment", "i like this!"], 
        "negative": ["this stinks", "you suck"],
    }
    akin = AkinClassifier(examples=examples)
    df = pd.read_csv("<some>/<file>.csv")

    # Calculate distances for the original dataframe
    akin.assign_distances(df)

    # Predict a single item
    akin.predict_single(text="thanks, that's nice of you")

    # Construct a generator that yields the {text, distances} dictionary for each item
    g = akin.pipe(text=df["text"])
    next(g)
    ```
    """
    def __init__(self, examples, featurizer=None, metric="euclidean", reducer="min"):
        self.featurizer = featurizer
        self.metric = metric
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

    def predict_single(self, text):
        """
        Predict a single text item.

        Arguments:
        - `text`: string to calculate distances for

        Usage:

        ```python
        import pandas as pd 
        from akin import AkinClassifier

        examples = {
            "positive": ["thanks so much", "compliment", "i like this!"], 
            "negative": ["this stinks", "you suck"],
        }
        akin = AkinClassifier(examples=examples)
        df = pd.read_csv("<some>/<file>.csv")

        # Predict a single item
        akin.predict_single(text="thanks, that's nice of you")
        ```
        """
        X_tfm = self.featurizer.transform([text])
        reducer_func = np.mean if self.reducer == "mean" else np.min
        return {k: reducer_func(pairwise_distances(X_tfm, self.examples_tfm[k], metric=self.metric)) for k in self.examples}
    
    def pipe(self, stream):
        """
        Predict a single text item.

        Arguments:
        - `stream`: stream of texts to add calculation for

        Usage:

        ```python
        import pandas as pd 
        from akin import AkinClassifier

        examples = {
            "positive": ["thanks so much", "compliment", "i like this!"], 
            "negative": ["this stinks", "you suck"],
        }
        akin = AkinClassifier(examples=examples)
        df = pd.read_csv("<some>/<file>.csv")

        # Construct a generator that yields the {text, distances} dictionary for each item
        g = akin.pipe(text=df["text"])
        next(g)
        ```
        """
        for item in stream:
            yield {"text": item, "distances": {**self.predict_single(item)}}
    
    def assign_distances(self, dataf, text_col="text"):
        """
        Predict a single text item.

        Arguments:
        - `dataf`: the dataframe in question
        - `text_col`: the column that contains the text data, assumed to be `text` by default

        Usage:

        ```python
        import pandas as pd 
        from akin import AkinClassifier

        examples = {
            "positive": ["thanks so much", "compliment", "i like this!"], 
            "negative": ["this stinks", "you suck"],
        }
        akin = AkinClassifier(examples=examples)
        df = pd.read_csv("<some>/<file>.csv")

        # Calculate distances for the original dataframe
        akin.assign_distances(df)
        ```
        """
        distances = [_['distances'] for _ in self.pipe(dataf[text_col])]
        return pd.concat([dataf, pd.DataFrame(distances)], axis=1)


def sort_dataframe(dataf, examples, featurizer=None, text_col="text", dist_col="dist", metric="euclidean", reducer="min"):
    """
    Sorts a dataframe based on the feature distances to the examples.

    Arguments:
    - `dataf`: a pandas dataframe
    - `examples`: list of text examples to compare against
    - `featurizer`: a sklearn compatible featurizer, if none is given we use a set of counvectorizers
    - `text_col`: the name of the text column in the dataframe to compare against
    - `dist_col`: the name of the column to store the calculated distance in
    - `metric`: the metric to use for distance calculation, default is "euclidean"
    - `reducer`: the method to use to reduce the distances of all examples to single text, default is "min"

    More information on the distance metrics can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html).
    
    Note that this function can also be used in a pandas pipeline. 

    ```python
    from akin import sort_dataframe 

    dataf = pd.read_csv("data.csv")
    dataf.pipe(sort_dataframe, examples=["very nice", "super positive"])
    ```
    """
    if not featurizer:
        featurizer = make_union(
            CountVectorizer(), 
            CountVectorizer(analyzer="char", ngram_range=(2, 3))
        )
    X_tfm = featurizer.fit_transform(examples)
    X_other = featurizer.transform(list(dataf[text_col]))
    distances = pairwise_distances(X_tfm, X_other, metric=metric)
    aggregated = distances.mean(axis=0) if reducer == "mean" else distances.min(axis=0)
    return dataf.assign(**{dist_col: aggregated}).sort_values("dist")
