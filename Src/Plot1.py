import numpy as np
import pandas as pd
import math
#import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import time
import sys
from cleanData import *

sc, st = clean()

def GenderPlot(st):
    male=sum(st['X1SEX']==1)
    female=sum(st['X1SEX']==2)
    barwidth=1

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

def IfPublic(st):

    indices = st[st.X1SCIEFF==-2.91].index
    df1 = st.drop(indices)
    public=sum(df1['X1CONTROL']==1)
    private=sum(df1['X1CONTROL']==2)
    barwidth=1

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

    plt.pie([public,private],labels=['Public','Non-Public'],autopct=lambda pct: func(pct, [public,private]),
            colors=['#2ECC71','#85C1E9'],startangle=90,textprops={'fontsize': 25})
    # plt.savefig(r"public_private.jpg")
    plt.show()
    #plt.bar([1],male,color='#2ECC71',
    #           width=barwidth,edgecolor='white',label='Male')
    #plt.bar([2],female,color='#85C1E9',
    #           width=barwidth,edgecolor='white',label='Female')
    cat1 = df1[df1['X1CONTROL']==1.0].X1SCIEFF #-sci_mean
    cat2 = df1[df1['X1CONTROL']==2.0].X1SCIEFF #-sci_mean

    stats.ttest_ind(cat1, cat2,equal_var=False)
    colors=['#2ECC71','#85C1E9']
    sns.set_palette(sns.color_palette(colors))
    ax = sns.boxplot(x="X1CONTROL", y="X1SCIEFF", data=df1)
    plt.ylabel('Science Self-Efficacy',fontsize='xx-large')
    plt.xticks([0,1],('Public','Non-public'),fontsize='xx-large')

    plt.savefig(r"public_private_eff.jpg")

    plt.show()

def SchoolRegion(sc):


    ##@title Region Graphs
    # print(set(list(sc_x1['X1REGION']))) #Northeast, Midwest, South, West
    r_pu=[] #public
    r_pr=[] #private
    r_c=[] #city
    r_s=[] #suburban
    r_t=[] #town
    r_r=[] #rural
    #Ref: https://nces.ed.gov/programs/edge/docs/LOCALE_CLASSIFICATIONS.pdf
    for i in [1,2,3,4]: #region
        r_pu.append(len(sc.query('X1REGION==%d and X1CONTROL==1' %(i))))
        r_pr.append(len(sc.query('X1REGION==%d and X1CONTROL==2' %(i))))
        r_c.append(len(sc.query('X1REGION==%d and X1LOCALE==1' %(i))))
        r_s.append(len(sc.query('X1REGION==%d and X1LOCALE==2' %(i))))
        r_t.append(len(sc.query('X1REGION==%d and X1LOCALE==3' %(i))))
        r_r.append(len(sc.query('X1REGION==%d and X1LOCALE==4' %(i))))

    bar1=[1,2,3,4]
    barwidth=0.5
    p1=plt.bar(bar1,r_pu,color='#2ECC71',
               width=barwidth,edgecolor='white',label='Public')
    p2=plt.bar(bar1,r_pr,bottom=r_pu,color='#85C1E9',
               width=barwidth,edgecolor='white',label='Private')
    plt.ylabel("Count",fontsize='xx-large')
    plt.title('School System by Region',fontsize='xx-large')
    plt.xticks(bar1,('Northeast','Midwest','South','West'),fontsize='xx-large')
    plt.yticks(np.arange(0,401,50),fontsize='xx-large')
    plt.legend((p2[0],p1[0]),('Private','Public'),fontsize='xx-large')
    plt.savefig(r"School_Type.png")
    plt.text(5,5,'Total=944')
    plt.show()

    barwidth=0.2
    r1=plt.bar(bar1,r_c,color='#F4D03F',
                width=barwidth,edgecolor='white',label='Northeast')
    r2=plt.bar([x + barwidth for x in bar1],r_s,color='#2ECC71',
                width=barwidth,edgecolor='white',label='Midwest')
    r3=plt.bar([x + barwidth*2 for x in bar1],r_t,color='#5DADE2',
                width=barwidth,edgecolor='white',label='South')
    r4=plt.bar([x + barwidth*3 for x in bar1],r_r,color='#BB8FCE',
                width=barwidth,edgecolor='white',label='West')

    plt.ylabel("Count",fontsize='xx-large')
    plt.title('School Urbanicity By Region',fontsize='xx-large')
    plt.xticks([x + barwidth*1.5 for x in bar1],('Northeast','Midwest','South','West'),fontsize='xx-large')
    plt.yticks(np.arange(0,250,50),fontsize='xx-large')
    plt.legend((r1[0],r2[0],r3[0],r4[0]),('City','Suburban','Town','Rural'),fontsize='xx-large')
    plt.savefig("School_Region.png")
    plt.show()

    plt.bar([x for x in range(1,5)],[sum(r_c),sum(r_s),sum(r_t),sum(r_r)],width=barwidth+0.5,edgecolor='white',color=["#F4D03F","#2ECC71","#5DADE2","#BB8FCE"])
    # sns.set_palette(sns.color_palette(colors))
    plt.xticks(bar1,('City','Suburban','Town','Rural'),fontsize='xx-large')
    plt.title('Number of School By Region',fontsize='xx-large')

def GenderEfficacyImpact(st):

    df = st[['STU_ID','X1SEX','X1SES', 'X1RACE','X1SCIEFF','X4REGION','X1CONTROL', 'X1LOCALE','N1SEX', 'X1TSCERT','N1COURSE']]
    df = df.astype({'X1SEX' : 'category','X1RACE': 'category','X4REGION':'category','X1CONTROL':'category','X1LOCALE':'category','N1SEX':'category','X1TSCERT':'category','N1COURSE':'category' })
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

def SciCourseImpact(st):


    df = st[['STU_ID','X1SEX','X1SES', 'X1RACE','X1SCIEFF','X4REGION','X1CONTROL', 'X1LOCALE','N1SEX', 'X1TSCERT','N1COURSE']]
    df = df.astype({'X1SEX' : 'category','X1RACE': 'category','X4REGION':'category','X1CONTROL':'category','X1LOCALE':'category','N1SEX':'category','X1TSCERT':'category','N1COURSE':'category' })
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
        if row['N1COURSE'] in LIFESCI:
            return 'LIFESCI'
        elif row['N1COURSE'] in EARTHSCI:
            return 'EARTHSCI'
        elif row['N1COURSE'] in PHYSSCI:
            return 'PHYSSCI'
        else:
            return 'OSCI'
        
    df['COURSE_TYPE'] = df.apply(lambda row: get_course_category(row,LIFESCI,EARTHSCI, PHYSSCI), axis=1)
    df1['COURSE_TYPE'] = df1.apply(lambda row: get_course_category(row,LIFESCI,EARTHSCI, PHYSSCI), axis=1)

    indices = df[df.X1SCIEFF==-2.91].index
    df1 = df.drop(indices)
    df1.groupby('COURSE_TYPE')['X1SCIEFF'].mean()
    val_cnts = df1['COURSE_TYPE'].value_counts().reindex(index=['PHYSSCI','OSCI','LIFESCI','EARTHSCI'])

    val_cnts.plot(kind = 'bar',rot='horizontal',color=["#F4D03F","#2ECC71","#5DADE2","#BB8FCE"])
    plt.xticks([i for i in range(4)],['Physical Science','Other Science','Life Science','Earth Science'],fontsize='xx-large')
    plt.title('Frequency of Science Course Taken by Students',fontsize='xx-large')

    plt.subplot(212)

    colors=["#F4D03F","#2ECC71","#5DADE2","#BB8FCE"]
    sns.set_palette(sns.color_palette(colors))
    stats.f_oneway(df1['X1SCIEFF'][df1['COURSE_TYPE']=='LIFESCI'],df1['X1SCIEFF'][df1['COURSE_TYPE']=='EARTHSCI'],
                  df1['X1SCIEFF'][df1['COURSE_TYPE']=='PHYSSCI'],df1['X1SCIEFF'][df1['COURSE_TYPE']=='OSCI'])
    plt.rcParams["figure.figsize"] = (15,10) #Resize image size
    ax = sns.boxplot(x="COURSE_TYPE", y="X1SCIEFF", data=df1)
    plt.xticks([i for i in range(4)],['Physical Science','Other Science','Life Science','Earth Science'],fontsize='xx-large')
    plt.ylabel('Science Self-Efficacy',fontsize='xx-large')
    plt.xlabel('Science Courses',fontsize='xx-large')

def EfficacyCorrelation(st):

    df_st = st[['X1SES','X1SCIEFF','X1SCIINT','X1SCIID','X1SCIUTI','S1TEFRNDS','S1TEACTIV','S1TEPOPULAR','S1TEMAKEFUN']]
    indices = df_st[df_st.X1SCIEFF==-2.91].index

    col_names = ['X1SES','X1SCIINT','X1SCIID','X1SCIUTI','S1TEFRNDS', 'S1TEACTIV', 'S1TEPOPULAR', 'S1TEMAKEFUN']

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

def AFFATT(st):

    # Create AFFATT variable - See basic stats of this, w.r.t teacher's sex and student sex. 
    # Use this variable to analyse science self-efficacy.
    df = st[['X1SEX','N1SEX','X1TSCERT','X1SCIEFF','N1GROUP','S1STCHVALUES','S1STCHRESPCT','S1STCHFAIR','S1STCHCONF',
                       'S1STCHMISTKE','S1STCHTREAT','S1STCHMFDIFF']]

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

def GenderAndCertif(st):
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

# Utils
def parentImpact(st):

    data = {};
    datalen = 23503;
    ParentInfluence = st[['STU_ID','X1SEX','X1SES', 'X1RACE','X1SCIEFF','X4REGION', 'P1MARSTAT', \
                          'P1HOMELANG', 'P1ELLEVER','P1HIDEG1','P1HIDEG2', 'P1JOBNOW1', 'P1JOBNOW2', 'P1INCOMECAT', 'P1ESTCONF']]
    
    temp = [ParentInfluence[(ParentInfluence['P1ELLEVER']==i+1)]['X1SCIEFF'].mean() for i in range(3)]
    data["ESL"] =  temp;
    temp = [ParentInfluence[(ParentInfluence['P1HIDEG1']==i)]['X1SCIEFF'].mean() for i in [1, 2, 3, 4, 5, 7 ]]
    data["Parent1Degree"] =  temp;
    temp = [ParentInfluence[(ParentInfluence['P1HIDEG2']==i)]['X1SCIEFF'].mean() for i in [1, 2, 3, 4, 5, 7 ]]
    data["Parent2Degree"] =  temp;



    temp = [ParentInfluence[(ParentInfluence['P1JOBNOW1']==i)]['X1SCIEFF'].mean() for i in range(2)]
    data["Job1"] =  temp;
    temp = [ParentInfluence[(ParentInfluence['P1JOBNOW2']==i)]['X1SCIEFF'].mean() for i in range(2)]
    data["Job2"] =  temp;


    temp = [ParentInfluence[(ParentInfluence['P1INCOMECAT']==i + 1)]['X1SCIEFF'].mean() for i in range(13)]

    data["Income"] =  temp;

    temp = [ParentInfluence[(ParentInfluence['P1ESTCONF']==i + 1)]['X1SCIEFF'].mean() for i in range(3)]

    data["CollegeTuition"] =  temp;

    return data

def ParentDeg(st):

    ####################################

    # Higher degree of any one of the parent will give obvious advantages for students on acience efficacy.
    # Parent 1 degree's influence is obviously larger than parent 2

    data = parentImpact(st)

    bar1=[1,2,3,4, 5, 6]

    mi = min(data["Parent1Degree"] + data["Parent2Degree"])
    inp = [x - mi for x in data["Parent1Degree"]]
    inp2 = [x - mi for x in data["Parent2Degree"]]

    barwidth=0.2

    r1=plt.bar(bar1,inp, width=barwidth, tick_label = ["Less", "High School", "Associate", "Bachelor", "Master", "PhD"])
    r2=plt.bar([x + barwidth for x in bar1],inp2,color='#fcdb03', width=barwidth)
    plt.legend((r1[0],r2[0]),("Parent 1", "Parent 2"),fontsize='xx-large')
    plt.ylabel("Average Science Efficacy Impact",fontsize='xx-large')

def ParentFactor(st):
    data = parentImpact(st)
    ###################################
    #Native speaker, parent 1 and 2's job status, and confidence on college tuition
    li = []

    for i in range(2):
      tem = []

      tem.append(data["ESL"][i])
      tem.append(data["Job1"][i]);
      tem.append(data["Job2"][i]);
      tem.append(data["CollegeTuition"][i]);
      li.append(tem);


    mi = min( min(li[0]), min(li[1]))

    bar1=[ 1,  2, 3, 4]
    mi = min( min(li[0]), min(li[1]))

    inp = [x - mi for x in li[0]]

    barwidth=0.2

    r1=plt.bar(bar1,inp, width=barwidth,  color = "red" , tick_label = ["Native English speaker",  "Parent 1 has Job", "Parent 2 has job", "Confidence on college tuition"]);

    inp = [x - mi for x in li[1]]
    r2=plt.bar([x + barwidth for x in bar1],inp, color = "lime" ,width=barwidth);

    #  tick_label = ["English as second Language student",  "Local")

    plt.legend((r1[0],r2[0]),("No", "Yes"),fontsize='xx-large')
    plt.xlabel("Impact factors",fontsize='xx-large')
    plt.ylabel("Average Science Efficacy Impact",fontsize='xx-large')

def ParentIncome(st):
    data = parentImpact(st)
    ###########################
    # Household income impact on student science efficacy.
    mi = min( data["Income"])

    inp = [x - mi for x in data["Income"]]

    barwidth=0.4

    plt.xlabel("Income upto (1,000 $)",fontsize='xx-large')
    r1=plt.bar(range(13),inp, color = "green", width=barwidth, tick_label = [15, 35, 55, 75, 95, 115, 135, 155, 175, 195, 215, 235, "Unlimited"])
    plt.ylabel("Average Science Efficacy Impact",fontsize='xx-large')
