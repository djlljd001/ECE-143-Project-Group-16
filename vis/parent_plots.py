import matplotlib.pyplot as plt
from src.preprocess import *
import pandas as pd

def parent_impact(st):
    '''
    This method computes the impact of the parent variables
    on science self-efficacy.
    :param st: Student Dataframe object
    :return: Science efficacy corresponding to parent variables
    '''
    assert isinstance(st, pd.DataFrame)

    data = {}
    ParentInfluence = st[['STU_ID', 'X1SEX', 'X1SES', 'X1RACE', 'X1SCIEFF', 'X4REGION',
                          'P1MARSTAT', 'P1HOMELANG', 'P1ELLEVER', 'P1HIDEG1', 'P1HIDEG2',
                          'P1JOBNOW1', 'P1JOBNOW2', 'P1INCOMECAT', 'P1ESTCONF']]

    temp = [ParentInfluence[(ParentInfluence['P1ELLEVER'] == i + 1)]['X1SCIEFF'].mean() for i in range(3)]
    data["ESL"] = temp
    temp = [ParentInfluence[(ParentInfluence['P1HIDEG1'] == i)]['X1SCIEFF'].mean() for i in [1, 2, 3, 4, 5, 7]]
    data["Parent1Degree"] = temp
    temp = [ParentInfluence[(ParentInfluence['P1HIDEG2'] == i)]['X1SCIEFF'].mean() for i in [1, 2, 3, 4, 5, 7]]
    data["Parent2Degree"] = temp

    temp = [ParentInfluence[(ParentInfluence['P1JOBNOW1'] == i)]['X1SCIEFF'].mean() for i in range(2)]
    data["Job1"] = temp
    temp = [ParentInfluence[(ParentInfluence['P1JOBNOW2'] == i)]['X1SCIEFF'].mean() for i in range(2)]
    data["Job2"] = temp

    temp = [ParentInfluence[(ParentInfluence['P1INCOMECAT'] == i + 1)]['X1SCIEFF'].mean() for i in range(13)]

    data["Income"] = temp

    temp = [ParentInfluence[(ParentInfluence['P1ESTCONF'] == i + 1)]['X1SCIEFF'].mean() for i in range(3)]

    data["CollegeTuition"] = temp

    return data


def parent_deg(st):
    '''
    This method plots the influence of the parents' highest degree
    on the student's science self-efficacy at various levels of
    degree certification of the parents. We observe,
    Higher degree of any one of the parent will give obvious
    advantages for students on science efficacy.
    Parent 1 degree's influence is obviously larger than parent 2.
    :param st: Student variable dataframe object.
    '''
    assert isinstance(st, pd.DataFrame)

    data = parent_impact(st)

    bar1 = [1, 2, 3, 4, 5, 6]

    mi = min(data["Parent1Degree"] + data["Parent2Degree"])
    inp = [x - mi for x in data["Parent1Degree"]]
    inp2 = [x - mi for x in data["Parent2Degree"]]

    barwidth = 0.2

    r1 = plt.bar(bar1, inp, width=barwidth,
                 tick_label=["Less", "High School", "Associate", "Bachelor", "Master", "PhD"])
    r2 = plt.bar([x + barwidth for x in bar1], inp2, color='#fcdb03', width=barwidth)
    plt.legend((r1[0], r2[0]), ("Parent 1", "Parent 2"), fontsize='xx-large')
    plt.ylabel("Average Science Efficacy Impact", fontsize='xx-large')


def parent_factor(st):
    '''
    This method plots various parent based variables that affect science
    self efficacy for a comparison between them. Plots the influence
    of parent's native language, parent's job status, and confidence
    in college tuition payability on student's science self-efficacy.
    :param st: Student variable dataframe object.
    '''
    assert isinstance(st, pd.DataFrame)

    data = parent_impact(st)
    li = []

    for i in range(2):
        tem = []

        tem.append(data["ESL"][i])
        tem.append(data["Job1"][i]);
        tem.append(data["Job2"][i]);
        tem.append(data["CollegeTuition"][i]);
        li.append(tem);

    mi = min(min(li[0]), min(li[1]))

    bar1 = [1, 2, 3, 4]
    mi = min(min(li[0]), min(li[1]))

    inp = [x - mi for x in li[0]]

    barwidth = 0.2

    r1 = plt.bar(bar1, inp, width=barwidth, color="red",
                 tick_label=["Native English speaker", "Parent 1 has Job", "Parent 2 has job",
                             "Confidence on college tuition"])

    inp = [x - mi for x in li[1]]
    r2 = plt.bar([x + barwidth for x in bar1], inp, color="lime", width=barwidth)

    plt.legend((r1[0], r2[0]), ("No", "Yes"), fontsize='xx-large')
    plt.xlabel("Impact factors", fontsize='xx-large')
    plt.ylabel("Average Science Efficacy Impact", fontsize='xx-large')


def parent_income(st):
    '''
    This method plots parents' household income on
    the students' science self-efficacy.
    :param st: Student variable Dataframe object
    '''
    assert isinstance(st, pd.DataFrame)

    data = parent_impact(st)
    mi = min(data["Income"])

    inp = [x - mi for x in data["Income"]]

    barwidth = 0.4

    plt.xlabel("Income upto (1,000 $)", fontsize='xx-large')
    r1 = plt.bar(range(13), inp, color="green", width=barwidth,
                 tick_label=[15, 35, 55, 75, 95, 115, 135, 155, 175, 195, 215, 235, "Unlimited"])
    plt.ylabel("Average Science Efficacy Impact", fontsize='xx-large')

if __name__ == "__main__":

    # Load the data into the school and student variables respectively.
    parser = argparse.ArgumentParser()
    parser.add_argument('school_file', type=str, default='data/hsls_school_v1_0.csv',
                        help='Path to file containing the school data')
    parser.add_argument('student_file', type=str, default='data/hsls_school_v1_0.csv',
                        help='Path to file containing the student data')
    args = parser.parse_args()

    _, st = clean(args.school_file, args.student_file)

    # Comment based on desired plots. Leave as is for all plots.
    parent_deg(st)
    parent_factor(st)
    parent_income(st)


