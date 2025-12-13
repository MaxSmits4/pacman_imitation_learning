import pickle
import torch

from data import state_to_tensor, INDEX_TO_ACTION
from architecture import PacmanNetwork


class SubmissionWriter:
    def __init__(self, test_set_path, model_path):
        """
        Initialize the writing of your submission.
        Pay attention that the test set only contains GameState objects,
        it's no longer (GameState, action) pairs.

        Arguments:
            test_set_path: The file path to the pickled test set.
            model_path: The file path to the trained model.
        """
        with open(test_set_path, "rb") as f:
            self.test_set = pickle.load(f)

        self.model = PacmanNetwork()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()


    def predict_on_testset(self):
        """
        Generate predictions for the test set.

        !!! Your predicted actions should follow the same order
        as the test set provided.
        """
        actions = []
        for state in self.test_set:
            x = state_to_tensor(state).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(x)
                pred_index = logits.argmax(dim=1).item()
                action = INDEX_TO_ACTION[pred_index]
                actions.append(action)
        return actions

    def write_csv(self, actions, file_name="submission"):
        """
        ! Do not modify !

        Write the predicted actions (North, South, ...)
        to a CSV file.

        """
        with open(file_name + ".csv", "w") as f:
            f.write("ACTION\n")  # Header
            for action in actions:
                f.write(f"{action}\n")


if __name__ == "__main__":
    writer = SubmissionWriter(
    test_set_path="datasets/pacman_test.pkl",
    model_path="pacman_model.pth"
    )
    predictions = writer.predict_on_testset()
    writer.write_csv(predictions)
