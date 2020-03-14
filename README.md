# Analysis of Ninth Grader’s Science Self Efficacy (ECE143 - Team 16)

## Team Members
- Shreyas Rajesh(A53324553)
- Subrato Chakravorty
- Kalvin Goode (A12620672)
- Jinglong Du (A12620446)

## Problem
Analysing students' self-efficacy(interest in taking up STEM subjects as their career) in STEM course during high school years.
In our case, to limit the extent of our study, we focus only on 9th Graders and Science self-efficacy.

## Dataset

We are using 2009 high school student and school data from the following website:


## Summary
We analyse the data and observe interesting correlation between various student, school and teacher level variables with science self-efficacy. Each of the variables were chosen from a large list of variables available in the dataset based on interesting results and variables with significant correlation.

1. At the student level, we consider student gender and choice of science course as indicators of science self-efficacy
2. At the school level, we consider the school ownership, region of US and type of locality to analyse the science self-efficacy
3. At the teacher level, we consider the affirmative attitude, teacher's gender and teacher's certification and study their impact on the science self-efficacy. 


## Applications
This work is extremely valuable in analysing the variables/factors that impact the science self-efficacy of students in the United States. Our strategies could be used by students while making decisions about their career as well as by teachers, parents and schools alike in order to determine what drives students towards certain disciplines and tailor their curriculum, pedagogy to utilise this to the fullest. Our work is also extremely scalable, to other class levels, other disciplines and even other countries, with extremely minimal tuning for those cases.  

## File Structure
```
Root
|
+----raw_data
|
+----
|
+----processed_data
|
+----scripts
|       |   plot1.py
|       |   cleanData.py
|
|    
|    main.py
|    Analysis.ipynb

```

## Instructions on running the code

* Python version: Python 3.6.6 64-bit
### Required packages

1. numpy
1. pandas
2. matplotlib

### Run the code

1. Download data from webside in order to import data. Note that our data is too large to upload to github. So please download it and put the data into the Data folder.
2. Import Plot1 as * shown in the Analysis.ipynb. It will automatically import data and get ready to plot.
3. Run each function for corresponding plots.

