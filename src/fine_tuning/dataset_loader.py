from datasets import load_dataset


class DatasetLoader:
    def __init__(self, hugging_face_df: str, device: str):
        """Loads Huggings Face dataset and formats it to pytorch with text and labels

        Args:
            hugging_face_df (str): Hugging Face dataset string
        """

        self.df = load_dataset(hugging_face_df)
        self.train, self.test = self.df["train"], self.df["test"]

        self.format_dataset(device=device)
        self.get_label_mapping()

    def format_dataset(self, device: str) -> None:
        """Formats dataset to pytorch and encode target"""
        self.train = self.train.class_encode_column("labels")
        self.test = self.test.class_encode_column("labels")

        self.train.set_format(type="torch", columns=["text", "labels"], device=device)
        self.test.set_format(type="torch", columns=["text", "labels"], device=device)

    def get_label_mapping(self) -> None:
        """Creates mapping of target labels and language names"""

        self.num_labels = len(self.train.features["labels"].names)
        self.id2label = {
            idx: self.train.features["labels"].int2str(idx)
            for idx in range(self.num_labels)
        }
        self.label2id = {label: idx for idx, label in self.id2label.items()}
