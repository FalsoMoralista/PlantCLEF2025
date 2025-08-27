# PlantCLEF 2025

<p align="">
  <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2005261%2Faa1c37e41dc33488cdc9747358d9e529%2FCapture%20decran%202025-01-13%20a%2014.37.20.png?generation=1736775460913154&alt=media">
</p>

This repository keeps the code required for attempting to **PlantClef 2025 challenge**.

Link to the [paper](https://www.dei.unipd.it/~faggioli/temp/clef2025/paper_241.pdf).

Challenge's details are available [here](https://www.kaggle.com/competitions/plantclef-2025).

---

# Proposal

The core of our method is a narrow vision transformer (ViT), which we employ to perform segmentation i.e., discriminate between plant (relevant) and background elements (irrelevant). This ViT is trained to reconstruct class prototypes (clusters) from the training set (containing individual plants) from high-resolution quadrat images (test set). We adapt this model to perform segmentation, by using its attention scores to eliminate the background elements, and then perform classification over the remaining relevant patches - in a decrease and conquer fashion. 

## Results

We were able to achieve challenge's **5th position** with a score of **0.33331**, only **0.03148** behind the first position.

---

# Usage

## Environment

Our whole project was built using **Python**, mainly **PyTorch**.

Required packages:

- PyTorch
- timm
- FAISS
- Pillow


## Scripts

We have splitted the training into three different scripts:

- `run_feature_extraction.sh`: extracts the features from the images containing a single plant sample
- `run_kmeans.sh`: clusters the extracted features into **7806 clusters**, one for each specie in the dataset
- `run_zero_shot.sh`: trains a narrow ViT to predict the target feature prototype vector

The scripts mentioned above should be called in the same order they are listed to achieve the required result.

### Tests

For the test step, the script `test_submission.sh` is responsible for loading the test dataset and using the trained ViT to segment the image into relevant and irrelevant patches based on attention, collate neighbors by proximity and classifying these aggregated patches.

## Data

All data used for training and tests are available at [challenge's page](https://www.kaggle.com/competitions/plantclef-2025/data).


---

# Acknowledgements

We would like to thanks Dr. Rodrigo Tripodi Calumby and Dr. Rodrigo Pereira David as our advisors. We also would like to thanks Lukáš Picek for support and encouragement.

This project was partially supported by UEFS and Capes.
