import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from src.preprocess import *
import pandas as pd

def if_public(st):
    '''
    This method plots the distribution of public and
    private schools based on region of the US.
    :param st: Student variables dataframe object.
    '''
    assert isinstance(st, pd.DataFrame)

    indices = st[st.X1SCIEFF==-2.91].index
    df1 = st.drop(indices)
    public=sum(df1['X1CONTROL']==1)
    private=sum(df1['X1CONTROL']==2)

    def func(pct, allvals):
        '''
            Provide percentage value and total value for each item.
            Return formatted value for pie chart to use
        '''
        isinstance(allvals,list)
        for i in allvals:
            pass
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.2f}%\n({:d})".format(pct, absolute)

    plt.pie([public,private],labels=['Public','Non-Public'],autopct=lambda pct: func(pct, [public,private]),
            colors=['#2ECC71','#85C1E9'],startangle=90,textprops={'fontsize': 25})
    plt.savefig(r"public_private.jpg")
    plt.show()

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

def school_region(sc):
    '''
    This method plots the locality and control of the school in
    various regions of the United States.
    :param sc: School variables dataframe object.
    '''
    assert isinstance(st, pd.DataFrame)

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

    plt.bar([x for x in range(1,5)],[sum(r_c),sum(r_s),sum(r_t),sum(r_r)],width=barwidth+0.5,
            edgecolor='white',color=["#F4D03F","#2ECC71","#5DADE2","#BB8FCE"])
    plt.xticks(bar1,('City','Suburban','Town','Rural'),fontsize='xx-large')
    plt.title('Number of School By Region',fontsize='xx-large')


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
    if_public(st)
    school_region(sc)
