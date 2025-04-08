# Introduction

### Exploring Movies Available on Netflix: Insights from TMDB Data

Welcome to my deep-dive analysis of **movies currently available on Netflix**, using a [large dataset](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates) I created from The Movie Database (TMDB). This dataset is updated daily and filtered to focus on titles that are **available on Netflix**, regardless of whether Netflix produced them.

### Context

Netflix has fundamentally transformed the entertainment industry, shifting the paradigm from theatrical-first to streaming-first. With a global subscriber base of over 300 million users and a yearly content budget surpassing \$17 billion, Netflix is not just a platform—it’s a global content distributor. While box office figures don’t fully reflect its performance, Netflix's content strategy blends original IP, local content, and global licensing deals.

### Objectives

This notebook aims to uncover content patterns, audience behavior, and creative signals from Netflix's current catalog through a series of data-driven visual explorations:

1. **Genre Trends**: Which genres dominate in volume, rating, and engagement?
2. **Origin Language**: How do content characteristics and performance vary by language?
3. **Runtime Optimization**: What runtimes correlate with better ratings and more viewer engagement?
4. **Content Performance Quadrants**: Clustering movies based on IMDb votes and rating to identify hits, hidden gems, and more.
5. **Talent Analysis**: Who are the top-performing directors, producers, writers, and composers based on audience engagement and critical acclaim?
6. **Production Companies**: Which companies consistently produce popular or highly-rated content on the platform?

### Note

TMDB is a community-driven platform, and its data may vary in completeness or accuracy—especially for streaming services like Netflix. The **revenue and budget data refer to theatrical box office metrics**, which are not relevant for most Netflix titles. As a result, this analysis focuses on more **actionable variables such as ratings, vote counts, runtime, origin, and talent**.


This notebook is built for content strategy exploration and does not cover marketing impact or subscriber growth, though these are key elements in Netflix's overall strategy.

<iframe src="charts/boxplot_genre_vote.html" width="100%" height="600px"></iframe>
