from akin import AkinClassifier


def test_base_clf():
    """Check that we cover an obvious usecase."""
    examples = {
        "positive": ["thanks so much", "compliment", "i like this!"],
        "negative": ["this stinks", "you suck"],
    }
    clf = AkinClassifier(examples, reducer="min")

    # Obvious positive examples
    distances1 = next(clf.pipe(["thanks a bunch"]))["distances"]
    assert distances1["positive"] <= distances1["negative"]

    # Obvious positive examples
    distances2 = next(clf.pipe(["you stink"]))["distances"]
    assert distances2["positive"] >= distances2["negative"]
