import pandas as pd

def readDataset(datasets):

    if datasets == 0:
        # antV0

        f1 = pd.read_csv("../final_dataset/ant/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/ant/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 1:
        # antV1

        f1 = pd.read_csv("../final_dataset/ant/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/ant/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 2:
        # antV2

        f1 = pd.read_csv("../final_dataset/ant/3.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/ant/4.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 3:
        # camelV0

        f1 = pd.read_csv("../final_dataset/camel/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/camel/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 4:
        # camelV1

        f1 = pd.read_csv("../final_dataset/camel/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/camel/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 5:
        # ivy

        f1 = pd.read_csv("../final_dataset/ivy/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/ivy/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 6:
        # jeditV0

        f1 = pd.read_csv("../final_dataset/jedit/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/jedit/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 7:
        # jeditV1

        f1 = pd.read_csv("../final_dataset/jedit/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/jedit/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 8:
        # jeditV2

        f1 = pd.read_csv("../final_dataset/jedit/3.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/jedit/4.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 9:
        # log4j

        f1 = pd.read_csv("../final_dataset/log4j/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/log4j/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 10:
        # lucene

        f1 = pd.read_csv("../final_dataset/lucene/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/lucene/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 11:
        # poiV0

        f1 = pd.read_csv("../final_dataset/poi/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/poi/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 12:
        # poiV1

        f1 = pd.read_csv("../final_dataset/poi/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/poi/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 13:
        # synapse

        f1 = pd.read_csv("../final_dataset/synapse/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/synapse/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 14:
        # velocity

        f1 = pd.read_csv("../final_dataset/velocity/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/velocity/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 15:
        # xercesV0

        f1 = pd.read_csv("../final_dataset/xerces/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/xerces/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 16:
        # xercesV1

        f1 = pd.read_csv("../final_dataset/xerces/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../final_dataset/xerces/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    return df1,df2