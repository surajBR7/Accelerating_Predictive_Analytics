# Accelerating Predictive Analytics on Large Datasets through Embarrassingly Parallel Computing

This project demonstrates how predictive analytics on large datasets can be accelerated using embarrassingly parallel computing techniques. The study uses a **GradientBoostingClassifier** to model data and applies parallel processing using Python's `multiprocessing` library to speed up computations.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data](#data)
- [Model](#model)
- [Parallel Processing](#parallel-processing)
- [Results](#results)
- [How to Run](#how-to-run)
- [References](#references)

## Overview
In this project, we investigate the tradeoff between data size and processing time. Predictive models often face challenges when the dataset size increases, leading to longer processing times. By using parallel computing, we can speed up the process of building and evaluating machine learning models.

## Project Structure
```plaintext
.
├── main.py              # Contains the main implementation of data preprocessing and model training
├── model.py             # Defines the GradientBoostingClassifier and parallel processing functions
├── Presentation.pptx    # Project presentation explaining methods and results
├── data                 # Folder for input data
├── README.md            # This file
```
## Data 

The dataset used contains 100,000 rows and 10 columns. After preprocessing, we selected four features: num_pages, book_rating, book_price, and text_lang. The target variable is book_genre.

To ensure that no single feature dominates the model, we used a standard scaler to normalize the data.

## Model
We utilized the GradientBoostingClassifier for classification, which builds decision trees sequentially. The settings used for training include:
  - n_estimators = 100
  - learning_rate = 0.1
    
The classifier was applied to the dataset and predictions were made based on the highest predicted probabilities.

## parallel Processing

### Serial Processing

In serial processing, tasks are executed one at a time. This leads to high workload for the processor and longer execution times.

### Parallel Processing

By using Python's multiprocessing library, we implemented parallel processing where multiple tasks are completed simultaneously across different processor cores. This approach significantly reduces the time required to process large datasets.

### Steps for Parallelization:
  - Import and preprocess the data.
  - Train the GradientBoostingClassifier model.
  - Apply multiprocessing with different configurations of cores and data sizes.
  - Measure time taken, speedup, and efficiency for each configuration.

## Result
Parallel processing demonstrated a noticeable speedup, especially for larger datasets. By distributing tasks across multiple cores, we reduced the time required to make predictions while maintaining model accuracy.

## How to run this:
- Clone the repository:
  ```bash
  git clone https://github.com/yourusername/your-repo-name.git
  cd your-repo-name
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
   ```
- Run the program:
  ```bash
  python3 main.py
  ```

## References
- [Python Multiprocessing Documentation](https://docs.python.org/3/library/multiprocessing.html)
- [Pandas Documentation](https://pandas.pydata.org/)
- [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [Towards Data Science: Parallelization in Python](https://towardsdatascience.com/parallelization-w-multiprocessing-in-python-bd2fc234f516)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

## Author
- Suraj Basavaraj Rajolad



  
