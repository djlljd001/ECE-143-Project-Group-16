import seaborn as sns
import matplotlib.pyplot as plt
from src.preprocess import *
import pandas as pd

# Load the data into the school and student variables respectively.
sc, st = clean()

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
    g = sns.FacetGrid(df, col="X1SEX")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('school_file', type=str, default='data/hsls_school_v1_0.csv',
                        help='Path to file containing the school data')
    parser.add_argument('student_file', type=str, default='data/hsls_school_v1_0.csv',
                        help='Path to file containing the student data')
    args = parser.parse_args()

    sc, st = clean(args.school_file, args.student_file)

    # Comment based on desired plots. Leave as is for all plots.
    affirmative_attitude(st)
    gender_certification(st)