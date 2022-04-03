<img src="akin.png" width=175 align="right">

# akin

> Some text similarity utilities

The goal of **akin** is to make it easy to sort text based on numeric similarity.

## Install 

You can install this tool via pip. 

```
python -m pip install "akin @ git+https://github.com/koaning/akin.git"
```

## Usage

The simplest way to use this tool is to just use it to sort texts. 

```python
from akin import sort_dataframe

# Let's load in a csv file that has a text column named "text". 
dataf = pd.read_csv("data.csv")
# Let's sort this dataframe such that we prefer examples with texts
# that are similar to the examples in the line below.
dataf.pipe(sort_dataframe, examples=["very nice", "super positive"], text_col="text")
```

In this basic setting, we're really just using CountVectors from scikit-learn
to compute the similarity between two texts based on bag of word counts. We could
go a bit more fancy though by using word embeddings from [whatlies](https://koaning.github.io/whatlies/tutorial/scikit-learn/). 
Our library supports any embedding, as long as it's implemented with the scikit-learn API
in mind.

```python
from whatlies.language import BytePairLanguage

bp_lang = BytePairLanguage("en")

dataf.pipe(sort_dataframe, 
           examples=["very nice", "super positive"], 
           text_col="text", 
           featurizer=bp_lang)
```

While the sorting will likely cover most activated labelling use-cases, you 
may also want an object that's a bit more flexible. For that you may use
the `AkinClassifier`.

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

## Warning 

I like to build in public but I should stress that this is a repo made for utility for myself. Honestly, it's made in a quick evening. Feel free to re-use, but don't expect maintenance or production-quality code in the long term.
