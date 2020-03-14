import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def teacher_gender(st):
    '''
    This method plots a pie chart of the number of 
    male and female teachers as well as the influence
    of gender on the science self-efficcay
    :param st: Student variables dataframe object
    '''

    indices = st[st.X1SCIEFF == -2.91].index
    df1 = st.drop(indices)
    public = sum(df1['N1SEX'] == 1)
    private = sum(df1['N1SEX'] == 2)

    def func(pct, allvals):
        '''
            Provide percentage value and total value for each item.
            Return formatted value for pie chart to use
        '''
        isinstance(allvals, list)
        for i in allvals:
            # isinstance(i>0)
            pass
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.2f}%\n({:d})".format(pct, absolute)

    plt.pie([public, private], labels=['Male Teachers', 'Female Teachers'],
            autopct=lambda pct: func(pct, [public, private]),
            colors=['#85C1E9', '#DC7633'], startangle=180, textprops={'fontsize': 25})
    plt.savefig(r"male_female_teach.jpg")
    plt.show()

    cat1 = df1[df1['N1SEX'] == 1].X1SCIEFF  # -sci_mean
    cat2 = df1[df1['N1SEX'] == 2].X1SCIEFF  # -sci_mean

    stats.ttest_ind(cat1, cat2, equal_var=False)
    colors = ['#85C1E9', '#DC7633']
    sns.set_palette(sns.color_palette(colors))
    ax = sns.boxplot(x="N1SEX", y="X1SCIEFF", data=df1)
    plt.ylabel('Student\'s Science Self-Efficacy', fontsize='xx-large')
    plt.xticks([0, 1], ('Male Teacher', 'Female Teacher'), fontsize='xx-large')
    plt.savefig(r"../images/male_female_teacheff.jpg")
    plt.show()


def teacher_certification(st):
    '''
    This method plots the impact of teacher_certification on the
    student's science self-efficacy.
    :param st: Student variables dataframe object.
    '''

    assert isinstance(st, pd.DataFrame)
    # Create AFFATT variable for analysus becuase of high correlation.
    indices = st[st.X1SCIEFF == -2.91].index
    df1 = st.drop(indices)
    #df1 = st[['X1SEX','N1SEX','X1TSCERT','X1SCIEFF','N1GROUP','S1STCHVALUES',
    #         'S1STCHRESPCT','S1STCHFAIR','S1STCHCONF','S1STCHMISTKE','S1STCHTREAT','S1STCHMFDIFF']]

    cat1 = df1[df1['X1TSCERT']==0].X1SCIEFF #-sci_mean
    cat2 = df1[df1['X1TSCERT']==1].X1SCIEFF #-sci_mean

    # indices = df1[df1.X1TSCERT.is[0,1]].index
    df2 = df1[df1.X1TSCERT.isin([0,1])]
    stats.ttest_ind(cat1, cat2,equal_var=False)
    colors=['#85C1E9','#2ECC71']
    sns.set_palette(sns.color_palette(colors))
    ax = sns.boxplot(x="X1TSCERT", y="X1SCIEFF", data=df2)
    plt.ylabel('Student\'s Science Self-Efficacy',fontsize='xx-large')
    plt.xticks([0,1],('No Regular Certification','Regular Certification'),fontsize='xx-large')
    plt.savefig(r"../images/teach_cert.jpg")
    # plt.ylim([-2.1,1.9])
    plt.show()

def affirmative_attitude(st):
    '''
    This method plots the variation of the affirmative attitude with the
    student's science self-efficacy.
    :param st: Student variables dataframe object.
    '''

    assert isinstance(st, pd.DataFrame)
    # Create AFFATT variable for analysus becuase of high correlation.
    df = st[['X1SEX','N1SEX','X1TSCERT','X1SCIEFF','N1GROUP','S1STCHVALUES',
             'S1STCHRESPCT','S1STCHFAIR','S1STCHCONF','S1STCHMISTKE','S1STCHTREAT','S1STCHMFDIFF']]

    df = df[df['X1SCIEFF'] != -2.91]
    df['S1STCHTREAT'] = st['S1STCHTREAT'].apply(lambda x: 5-x)
    df['S1STCHMFDIFF'] = st['S1STCHMFDIFF'].apply(lambda x: 5-x)
    #df['AFFATT'] = round(df[['S1STCHVALUES','S1STCHRESPCT','S1STCHFAIR','S1STCHCONF',
    #                   'S1STCHMISTKE','S1STCHTREAT','S1STCHMFDIFF']].sum(axis=1)/7.0)
    df['AFFATT'] = df[['S1STCHVALUES','S1STCHRESPCT','S1STCHFAIR','S1STCHCONF',
                       'S1STCHMISTKE','S1STCHTREAT','S1STCHMFDIFF']].sum(axis=1)/7.0
    df['AFFATT'] = df['AFFATT'].apply(lambda x: 5-x)
    print(df.head())
    sns.set(style="darkgrid")
    plt.rcParams["figure.figsize"] = (25,25)
    sns.lineplot(y='X1SCIEFF', x='AFFATT',data=df)


def linear_regression_affatt(st):
    '''
    This method performs a linear regression analysis of the teacher's
    attitude variables against student's science self-efficacy
    showing us that affirmative attitude is the largest influencer of
    student science self-efficacy.
    :param st: Student variables dataframe object
    '''
    assert isinstance(st, pd.DataFrame)
    # Linear regression only on teacher variables
    plt.rcParams["figure.figsize"] = (10,10)
    df = st[['X1SEX','N1INTEREST','N1CONCEPTS','N1TEST','N1PREPARE','N1IDEAS','N1SEX','X1TSCERT','X1SCIEFF','N1GROUP','S1STCHVALUES','S1STCHRESPCT','S1STCHFAIR','S1STCHCONF',
                       'S1STCHMISTKE','S1STCHTREAT','S1STCHMFDIFF']]

    df = df[df['X1SCIEFF'] != -2.91]
    df['S1STCHTREAT'] = st['S1STCHTREAT'].apply(lambda x: 5-x)
    df['S1STCHMFDIFF'] = st['S1STCHMFDIFF'].apply(lambda x: 5-x)

    df['AFFATT'] = df[['S1STCHVALUES','S1STCHRESPCT','S1STCHFAIR','S1STCHCONF',
                       'S1STCHMISTKE','S1STCHTREAT','S1STCHMFDIFF']].sum(axis=1)/7.0
    df['AFFATT'] = df['AFFATT'].apply(lambda x: 5-x)
    final_df = df[['N1INTEREST','N1TEST','N1CONCEPTS','N1PREPARE','N1IDEAS','N1SEX','X1TSCERT','N1GROUP','AFFATT']]
    final_df = final_df.astype('category')

    model = LinearRegression()
    final_df.head()
    model.fit(final_df, df['X1SCIEFF'])
    model.score(final_df, df['X1SCIEFF'])
    Y_pred = model.predict(final_df)
    coefs = dict(zip(final_df.columns,model.coef_))
    for key, val in coefs.items():
        print('{} : {}'.format(key, val))
    # model.coef_
    #print(type(model.coef_))
    #print(coefs.keys)

    # Plotting
    colors=["#5DADE2"]
    sns.set_style("whitegrid")
    #sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.set_palette(sns.color_palette(colors))
    plt.barh(list(coefs.keys()), list(coefs.values()))
    plt.yticks([x for x in range(9)],['Interest', 'Preparation', 'Concepts', 'Certification', 'Teacher\'s gender',\
                                      'Test Quality', 'Affirmative attitude', 'New Ideas', 'Group Work'],fontsize='xx-large')
    plt.xlim([-0.1,0.4])
    plt.xlabel('Correlation with Science Self-Efficacy',fontsize='xx-large')


def affatt_boxplots(st):
    '''
    This method compares the influence of teacher's affirmative attitude on the
    different genders of student and plots the variation between the two..
    :param st: Student variables dataframe object.
    '''
    assert isinstance(st, pd.DataFrame)
    # Create AFFATT variable for analysus becuase of high correlation.
    df = st[['X1SEX', 'N1SEX', 'X1TSCERT', 'X1SCIEFF', 'N1GROUP', 'S1STCHVALUES',
             'S1STCHRESPCT', 'S1STCHFAIR', 'S1STCHCONF', 'S1STCHMISTKE', 'S1STCHTREAT', 'S1STCHMFDIFF']]
    df = df[df['X1SCIEFF'] != -2.91]
    df['S1STCHTREAT'] = st['S1STCHTREAT'].apply(lambda x: 5-x)
    df['S1STCHMFDIFF'] = st['S1STCHMFDIFF'].apply(lambda x: 5-x)

    # Boxplots for variation of sex with affirmative attitude
    df['AFFATT'] = round(df[['S1STCHVALUES', 'S1STCHRESPCT', 'S1STCHFAIR', 'S1STCHCONF',
                             'S1STCHMISTKE', 'S1STCHTREAT', 'S1STCHMFDIFF']].sum(axis=1) / 7.0)
    df['AFFATT'] = df['AFFATT'].apply(lambda x: 5 - x)
    plt.rcParams["figure.figsize"] = (5, 5)
    df_m = df[df['X1SEX'] == 1.0]
    df_f = df[df['X1SEX'] == 2.0]

    colors = ["#F4D03F", "#2ECC71", "#5DADE2", "#BB8FCE"]
    sns.set_style("whitegrid", {'axes.grid': False})
    f, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True)
    # sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.set_palette(sns.color_palette(colors))
    ax = sns.boxplot(x="AFFATT", y="X1SCIEFF", data=df_m, ax=axes[0])
    ax = sns.boxplot(x="AFFATT", y="X1SCIEFF", data=df_f, ax=axes[1])


def gender_certification(st):
    '''
    This method compares the teacher's gender and certification with the
    student's science self-efficacy.
    :param st: Student variables dataframe object.
    '''

    assert isinstance(st, pd.DataFrame)
    df = st[['STU_ID','X1SEX','X1SES', 'N1GROUP', 'N1SCIJOB', 'X1RACE','X1SCIEFF','N1SCIYRS912',
         'X4REGION','X1CONTROL', 'X1LOCALE','N1SEX', 'X1TSCERT']]

    male = df[df['N1SEX'] == 1]
    female = df[df['N1SEX'] == 2]
    cert = df[df['X1TSCERT'] == 1]
    noncert = df[df['X1TSCERT'] == 0]
    # plt.pie([len(male), len(female)], labels=['MALE', 'FEMALE'])
    # plt.figure()
    # plt.pie([len(cert), len(noncert)], labels=['CERTIFIED', 'NON-CERTIFIED'])
    male_cert = male[male['X1TSCERT'] == 1]
    female_cert = female[female['X1TSCERT'] == 1]
    male_noncert = male[male['X1TSCERT'] == 0]
    female_noncert = female[female['X1TSCERT'] == 0]
    flist = (len(male_cert), len(female_cert))
    p1 = plt.barh(np.arange(2),flist , height=0.75)
    p2 = plt.barh(np.arange(2), (len(male_noncert), len(female_noncert)),left=flist, height=0.75)
    plt.legend((p1, p2), ("CERT", "NO CERT"))

if __name__ == "__main__":

    # Load the data into the school and student variables respectively.
    import sys
    sys.path.append('../src/')
    from preprocess import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--school_file', type=str, required=False, default='../data/hsls_school_v1_0.csv',
                        help='Path to file containing the school data')
    parser.add_argument('--student_file', type=str, required=False, default='../data/hsls_school_v1_0.csv',
                        help='Path to file containing the student data')
    args = parser.parse_args()

    sc, st = clean(args.school_file, args.student_file)

    # Comment based on desired plots. Leave as is for all plots.
    linear_regression_affatt(st)
    affatt_boxplots(st)
    affirmative_attitude(st)
    gender_certification(st)
