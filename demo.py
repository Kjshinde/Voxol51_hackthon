import fiftyone as fo
import fiftyone.zoo as foz

# Load 50 samples from COCO validation set
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=50,
    shuffle=True,
)
