# Fitspiration Tweets Project

The code and data in this repository belong to a project for MACS 30200 "Perspectives on Computational Research" at the University of Chicago.

The code is written in Python 3.9.13 and all of its dependencies can be installed by running the following in the terminal (with the `requirements.txt` file included in this repository):

```
pip install -r requirements.txt
```

## Initial Findings
After conducting sentiment analysis using VADER, I found that the mean compound score of sentiment is highest for late-pandemic fitpiration tweets, followed by pre-pandemic and then early-pandemic (as shown in the plot below). However, there is no statistically significant difference between the sentiment of the three groups, which was confirmed with an ANOVA test.

![Alt Text](analysis/sentiment_scores.png)

This finding directly answers my research question regarding changes in sentiment expressed in fitspiration tweets during these three periods (the other piece of the puzzle being the topics). There may be a shift in the overall sentiment expressed in fitspiration tweets over time, with a more positive sentiment being expressed during the late-pandemic period, with early-pandemic tweets being the least optimistic. However, based on this sample, we are unable to conclude that there is a statistically significant difference between the sentiment expressed in the three periods.

## Replication
To replicate and produce the finding, people could run through `analyze.ipynb` under the directory `analyze`. Note that `clean_data.csv` under `data` directory is used.

## How to Cite
To cite this replication materials repository, please use the following format:
Li, J. (2022). Fitspiration Tweets Project. GitHub. https://github.com/macs30200-s23/fitspiration-tweets-project

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

## Brief Description of Folders
1. `raw`: This directory contains codes collecting raw data from Twitter.
2. `clean`: This directory contains codes pre-processing tweets.
3. `analysis`: This directory contains codes analyzing data.
4. `data`: This folder contains raw data and cleaned data. 
    - `raw_data.csv`: collected using snscrape. The data are in different formats and require cleaning before analysis.
    - `clean_data.csv` used for analysis. It is in a standardized format and ready to be fed into modeling or statistical analysis.
