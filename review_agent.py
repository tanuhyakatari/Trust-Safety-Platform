from pathlib import Path
import pickle


class ReviewAgent:

    def __init__(self):

        BASE_DIR = Path(__file__).resolve().parent.parent

        model_path = (
            BASE_DIR /
            "models" /
            "fake_review_model.pkl"
        )

        vectorizer_path = (
            BASE_DIR /
            "models" /
            "tfidf_vectorizer.pkl"
        )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)


    def predict(self, review):

        features = self.vectorizer.transform(
            [review]
        )

        prediction = self.model.predict(
            features
        )[0]

        probability = self.model.predict_proba(
            features
        )[0]

        confidence = max(probability) * 100


        if prediction == 1:
            result = "Genuine Review"
        else:
            result = "Fake Review"


        return {
            "prediction": result,
            "confidence": round(confidence, 2)
        }



if __name__ == "__main__":

    agent = ReviewAgent()

    review = input(
        "Enter review: "
    )

    result = agent.predict(review)

    print("\nResult:")
    print(result)