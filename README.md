# Analysis of Ninth Grader’s Science Self Efficacy (ECE143 - Team 16)

## Team Members
- Shreyas Rajesh (A53324553)
- Subrato Chakravorty (A53322325)
- Kalvin Goode (A12620672)
- Jinglong Du (A12620446)

## Problem
Analysing students' self-efficacy (interest in taking up STEM subjects as their career) in STEM course during high school years.
In our case, to limit the extent of our study, we focus only on 9th Graders and Science self-efficacy.

## Dataset

This study employs public-use data from the High School Longitudinal Study of 2009 (HSLS:09). One important difference
between HSLS:09 and previous studies is its focus on STEM education; one specific goal of the study is to gain an understanding of the factors that lead students to choose science, technology, engineering, and mathematics courses, majors, and careers. 

Dataset can be downloaded by clicking: https://nces.ed.gov/EDAT/Data/Zip/HSLS_2016_v1_0_CSV_Datasets.zip

Steps:
1. unzip Data/Zip/HSLS_2016_v1_0_CSV_Datasets.zip
2. Change the name of the file 'hsls_09_school_v1_0.csv' to 'hsls_school_v1_0.csv'
3. Change the name of the file 'hsls_16_student_v1_0.csv' to 'hsls_student_v1_0.csv'
4. Move both the files to Data folder inside the project.



## Summary
We analyse the data and observe interesting correlation between various student, school and teacher level variables with science self-efficacy. Each of the variables were chosen from a large list of variables available in the dataset based on interesting results and variables with significant correlation.

1. At the student level, we consider student gender and choice of science course as indicators of science self-efficacy
2. At the school level, we consider the school ownership, region of US and type of locality to analyse the science self-efficacy
3. At the teacher level, we consider the affirmative attitude, teacher's gender and teacher's certification and study their impact on the science self-efficacy. 
4. At the parent level, we consider the household income, parents' job status, degree earned, native english speaker family , confidence on college tuition and their impact on the science self-efficacy.

## Applications
This work can prove to be valuable in analysing the variables/factors that impact the science self-efficacy of students in the United States. Our strategies could be used by students while making decisions about their career as well as by teachers, parents and schools alike in order to determine what drives students towards certain disciplines and tailor their curriculum, pedagogy to utilise this to the fullest. Our work is also easily scalable, to other class levels, other disciplines and even other countries, with minimal tuning for those cases.  

## File Structure
```
Root
|
+----data
|      |  hsls_school_v1_0.csv
|      |  hsls_student_v1_0.csv
|
|
+----images
|
+----src
|     |   preprocess.py
|     |   Analysis.ipynb
|
+----vis
|     |   student_plots.py
|     |   school_plots.py
|     |   teacher_plots.py
|     |   parent_plots.py
|
+---- README.md

```

## Instructions on running the code

* Python version: Python 3.7.3 64-bit
### Required packages

1. numpy
1. pandas
2. matplotlib
3. seaborn
4. scikit-learn

### Run the code

1. Download data from website and follow steps as explained in the dataset subsection. Note that our data is too large to upload to github.
2. To obtain full analysis run Analysis.ipynb. It will automatically import data and generate all the plots for all our various experiments.
3. Alternatively, each plot file in the visualization folder as well as the preprocess file can be run independently.
4. To do so, we need to use the following commands from the terminal in the root folder. As an example,
```
cd vis
python student_plots.py --school_file=../data/hsls_school_v1_0.csv --student_file=../data/hsls_student_v1_0.csv
```
More generally, 
```
cd <path_to_folder_of_py_file>
python <path_to_py_file> --school_file=<path_to_school_data_file> --student_file=<path_to_student_data_file>
```
5. To understand the argument parameters better, one can run,
```
python <path_to_py_file> --help
```

