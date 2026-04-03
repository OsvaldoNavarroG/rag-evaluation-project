def evaluate_answer(predicted, expected):
    predicted = predicted.lower()
    expected = expected.lower()

    return expected in predicted