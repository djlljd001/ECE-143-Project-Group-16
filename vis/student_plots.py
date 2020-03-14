import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from src.preprocess import *
import pandas as pd

def gender_plot(st):
    '''
    This method gives us a pie chart of the distribution
    of student's gender in the dataset.
    :param st: Student variables dataframe object
    '''
    assert isinstance(st, pd.DataFrame)

    male=sum(st['X1SEX']==1)
    female=sum(st['X1SEX']==2)

    def func(pct, allvals):
        '''
            Provide percentage value and total value for each item.
            Return formatted value for pie chart to use
        '''
        isinstance(allvals,list)
        for i in allvals:
            #isinstance(i>0) 
            pass
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.2f}%\n({:d})".format(pct, absolute)

    plt.pie([male,female],labels=['Male','Female'],autopct=lambda pct: func(pct, [male,female]),
            colors=['#85C1E9','#DC7633'],startangle=90,textprops={'fontsize': 25})
    plt.savefig(r"Male_Female.jpg")
    plt.show()

def gender_efficacy_impact(st):
    '''
    This method plots the variation of science self-efficacy with gender.
    We observe significant differences between male and female students.
    :param st: Student variables dataframe object
    '''
    assert isinstance(st, pd.DataFrame)

    df = st[['STU_ID','X1SEX','X1SES', 'X1RACE','X1SCIEFF','X4REGION','X1CONTROL',
             'X1LOCALE','N1SEX', 'X1TSCERT','N1COURSE']]
    df = df.astype({'X1SEX' : 'category','X1RACE': 'category','X4REGION':'category',
                    'X1CONTROL':'category','X1LOCALE':'category','N1SEX':'category',
                    'X1TSCERT':'category','N1COURSE':'category' })
    df.dropna(inplace=True)
    indices = df[df.X1SCIEFF==-2.91].index
    df1 = df.drop(indices)

    sci_mean = df.X1SCIEFF.mean()
    cat1 = df[df['X1SEX']==1.0].X1SCIEFF #-sci_mean
    cat2 = df[df['X1SEX']==2.0].X1SCIEFF #-sci_mean

    stats.ttest_ind(cat1, cat2,equal_var=False)
    colors=["#85C1E9","#DC7633"]
    sns.set_palette(sns.color_palette(colors))
    ax = sns.boxplot(x="X1SEX", y="X1SCIEFF", data=df1)
    plt.ylabel('Science Self-Efficacy',fontsize='xx-large')
    plt.xticks([0,1],('Male','Female'),fontsize='xx-large')
    plt.show()

def sci_couse_impact(st):
    '''
    This method plots the impact of choice of science courses on
    the student's science self-efficacy.
    :param st: Student variables dataframe object
    '''
    assert isinstance(st, pd.DataFrame)

    df = st[['STU_ID','X1SEX','X1SES', 'X1RACE','X1SCIEFF','X4REGION','X1CONTROL',
             'X1LOCALE','N1SEX', 'X1TSCERT','N1COURSE']]
    df = df.astype({'X1SEX' : 'category','X1RACE': 'category','X4REGION':'category',
                    'X1CONTROL':'category','X1LOCALE':'category','N1SEX':'category',
                    'X1TSCERT':'category','N1COURSE':'category' })
    df.dropna(inplace=True)

    indices = df[df.X1SCIEFF==-2.91].index
    df1 = df.drop(indices)

    plt.subplot(211)

    df.N1COURSE.unique()
    plt.rcParams["figure.figsize"] = (10,10) #Resize image size

    # Course categories
    LIFESCI = [2,10,11,12,13] 
    EARTHSCI = [3,4,5]
    PHYSSCI = [6,7,8,14,15,16,17,21]

    def get_course_category(row, LIFESCI, EARTHSCI, PHYSSCI):
        '''
        This method determines the category of the course from the
        dataframe row.
        '''
        if row['N1COURSE'] in LIFESCI:
            return 'LIFESCI'
        elif row['N1COURSE'] in EARTHSCI:
            return 'EARTHSCI'
        elif row['N1COURSE'] in PHYSSCI:
            return 'PHYSSCI'
        else:
            return 'OSCI'
        
    df['COURSE_TYPE'] = df.apply(lambda row: get_course_category(row,LIFESCI,
                                                                 EARTHSCI, PHYSSCI), axis=1)
    df1['COURSE_TYPE'] = df1.apply(lambda row: get_course_category(row,LIFESCI,
                                                                   EARTHSCI, PHYSSCI), axis=1)

    indices = df[df.X1SCIEFF==-2.91].index
    df1 = df.drop(indices)
    df1.groupby('COURSE_TYPE')['X1SCIEFF'].mean()
    val_cnts = df1['COURSE_TYPE'].value_counts().reindex(index=['PHYSSCI','OSCI',
                                                                'LIFESCI','EARTHSCI'])

    val_cnts.plot(kind = 'bar',rot='horizontal',color=["#F4D03F","#2ECC71","#5DADE2","#BB8FCE"])
    plt.xticks([i for i in range(4)],['Physical Science','Other Science',
                                      'Life Science','Earth Science'],fontsize='xx-large')
    plt.title('Frequency of Science Course Taken by Students',fontsize='xx-large')

    plt.subplot(212)

    colors=["#F4D03F","#2ECC71","#5DADE2","#BB8FCE"]
    sns.set_palette(sns.color_palette(colors))
    stats.f_oneway(df1['X1SCIEFF'][df1['COURSE_TYPE']=='LIFESCI'],
                   df1['X1SCIEFF'][df1['COURSE_TYPE']=='EARTHSCI'],
                   df1['X1SCIEFF'][df1['COURSE_TYPE']=='PHYSSCI'],
                   df1['X1SCIEFF'][df1['COURSE_TYPE']=='OSCI'])
    plt.rcParams["figure.figsize"] = (15,10) #Resize image size
    ax = sns.boxplot(x="COURSE_TYPE", y="X1SCIEFF", data=df1)
    plt.xticks([i for i in range(4)],['Physical Science','Other Science',
                                      'Life Science','Earth Science'],fontsize='xx-large')
    plt.ylabel('Science Self-Efficacy',fontsize='xx-large')
    plt.xlabel('Science Courses',fontsize='xx-large')

def efficacy_correlation(st):
    '''
    This method uses a linear regression to analyse the impact
    of various student variables on their science self-efficacy
    and determine the most importatnt factors from it. It also
    plots the coefficients of each of the factors that affect
    science self-efficacy.
    the student's science self-efficacy.
    :param st: Student variables dataframe object
    '''
    assert isinstance(st, pd.DataFrame)

    df_st = st[['X1SES','X1SCIEFF','X1SCIINT','X1SCIID','X1SCIUTI',
                'S1TEFRNDS','S1TEACTIV','S1TEPOPULAR','S1TEMAKEFUN']]
    indices = df_st[df_st.X1SCIEFF==-2.91].index

    col_names = ['X1SES','X1SCIINT','X1SCIID','X1SCIUTI','S1TEFRNDS',
                 'S1TEACTIV', 'S1TEPOPULAR', 'S1TEMAKEFUN']

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(df_st[col_names], df_st['X1SCIEFF'])

    Y_pred = model.predict(df_st[col_names])

    print('')
    y = []
    for col in col_names[:4]:
        print("Column name: ",col)
        y.append(stats.pearsonr(df_st[col],df_st['X1SCIEFF'])[0])
    for col in col_names[4:]:
        y.append((stats.pearsonr(5 - df_st[col],df_st['X1SCIEFF'])[0]))
        print("Column name: ",col)
    print(y)
    colors=["#5DADE2"]
    # set_style("whitegrid")
    sns.set_palette(sns.color_palette(colors))
    plt.barh([x for x in range(8)],y)
    plt.yticks([x for x in range(8)],['Interest','Identity','Utility',
                                      'Friends','Activity','Popular','Ridicule'],fontsize='xx-large')
    plt.xlabel('Correlation with Science Self-Efficacy',fontsize='xx-large')


if __name__ == "__main__":

    # Load the data into the school and student variables respectively.
    parser = argparse.ArgumentParser()
    parser.add_argument('school_file', type=str, default='data/hsls_school_v1_0.csv',
                        help='Path to file containing the school data')
    parser.add_argument('student_file', type=str, default='data/hsls_school_v1_0.csv',
                        help='Path to file containing the student data')
    args = parser.parse_args()

    sc, st = clean(args.school_file, args.student_file)

    # Comment based on desired plots. Leave as is for all plots.
    gender_plot(st)
    gender_efficacy_impact(st)
    sci_couse_impact(st)
    efficacy_correlation(st)

