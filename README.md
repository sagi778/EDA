# Data_Explorer App

simple & fast Data Exploration tool,
it allows you to explore data along many files.
App is built on simple "Command blocks" - each block contain an analysis which is a peace in the puzzle.

# Preview blocks:
![get_preview](screenshots/get_preview.png)

![get_data](screenshots/get_data.png)

![get_columns_info](screenshots/get_columns_info.png)

![get_numerics_desc](screenshots/get_numerics_desc.png)

![get_categorical_desc](screenshots/get_categorical_desc.png)


# Visualization code blocks:
### get_dist_plot() : 
Explore Numeric values distribution with verious outliers options:
![get_dist_plot_iqr](screenshots/get_dist_plot_iqr.png)
![get_dist_plot_3perc](screenshots/get_dist_plot_3perc.png)

### get_relation_plot() : 
exploring relation between 2 numeric columns:
* include all data points (=no outliers):
![get_relation_plot_no_categories_no_outliers](screenshots/get_relation_plot_no_categories_no_outliers.png)
* include only inlier data points (= outliers excluded based on [LocalOutlierFactor](https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) density value):
![get_relation_plot_no_categories_5perc_outliers](screenshots/get_relation_plot_no_categories_5perc_outliers.png)
* break the data to categories:
![get_relation_plot_2_categories_5perc_outliers](screenshots/get_relation_plot_2_categories_5perc_outliers.png)



![get_dist_plot](screenshots/get_dist_plot.png)



![get_comapre](screenshots/get_compare.png)

# Visualization/Tabular code blocks:
  ## get_corralations() : 
  can be shown in tabular/chart - allows an overview of linear correlation magnitude(=r^2)
  ![get_correlations](screenshots/get_correlations.png)

# Contact me:
Help me improve,

see anything missing?

think that some function/code block might be helpful to you?

leave a comment 



