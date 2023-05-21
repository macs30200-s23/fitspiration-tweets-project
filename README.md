# Fitspiration Tweets Project

The code and data in this repository belong to a project for MACS 30200 "Perspectives on Computational Research" at the University of Chicago.

The code is written in Python 3.9.13 and all of its dependencies can be installed by running the following in the terminal (with the `requirements.txt` file included in this repository):

```
pip install -r requirements.txt
```

## Replication
To replicate and produce the finding, people could run through `sentiment_analysis.ipynb` and `volume_hashtag_tfidf_lda.ipynb` under the directory `analysis`.

## How to Cite
To cite this replication materials repository, please use the following format:
```
Li, J. (2022). Fitspiration Tweets Project. GitHub. https://github.com/macs30200-s23/fitspiration-tweets-project
```
In BibTeX:
```
@misc{jiayan2023,
  author = {Jiayan, Li},
  title = {Fitspiration Tweets Project},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/macs30200-s23/fitspiration-tweets-project}},
  commit = {3be475ad25d462db47878fb975398127fb14478e}
}
```

## Repository Structure
1. `raw`: This directory contains codes collecting raw data from Twitter.
2. `clean`: This directory contains codes pre-processing tweets.
3. `analysis`: This directory contains codes analyzing data.
  - `sentiment_analysis.ipynb` and `volume_hashtag_tfidf_lda.ipynb`: contains the complete code to get the full results in the paper.
  - `visualization`: contains visualization produced after running throught the two notebooks `sentiment_analysis.ipynb` and `volume_hashtag_tfidf_lda.ipynb`.
  - `model`: contains all of the LDA models built from running `volume_hashtag_tfidf_lda.ipynb`.
  - `results`: contains the representative tweets for each topic of the LDA results.
  - `analyze.py`: contains helper functions

4. `data`: This folder contains raw data and cleaned data. 
    - `raw_data.csv`: collected using snscrape. The data are in different formats and require cleaning before analysis.
    - `processed.csv` used for sentiment analysis. It is in a standardized format and ready to be fed into modeling or statistical analysis.
    - `sentiment.csv` is produced after running through `sentiment_analysis.ipynb` and used in `volume_hashtag_tfidf_lda.ipynb` for the rest of analysis.
5. `utils.py`: contains universal helper functions used in `raw`, `clean`, and `analysis`.