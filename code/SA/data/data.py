import pandas as pd


def readDataset(datasets):

    if datasets == 0:
        # antV0


        f1 = pd.read_csv("../../uncombined_dataset_modified/ant/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/ant/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 1:
        # antV1

        f1 = pd.read_csv("../../uncombined_dataset_modified/ant/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/ant/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 2:
        # antV2

        f1 = pd.read_csv("../../uncombined_dataset_modified/ant/3.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/ant/4.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 3:
        # camelV0

        f1 = pd.read_csv("../../uncombined_dataset_modified/camel/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/camel/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 4:
        # camelV1

        f1 = pd.read_csv("../../uncombined_dataset_modified/camel/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/camel/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 5:
        # ivy

        f1 = pd.read_csv("../../uncombined_dataset_modified/ivy/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/ivy/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 6:
        # jeditV0

        f1 = pd.read_csv("../../uncombined_dataset_modified/jedit/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/jedit/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 7:
        # jeditV1

        f1 = pd.read_csv("../../uncombined_dataset_modified/jedit/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/jedit/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 8:
        # jeditV2

        f1 = pd.read_csv("../../uncombined_dataset_modified/jedit/3.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/jedit/4.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 9:
        # log4j

        f1 = pd.read_csv("../../uncombined_dataset_modified/log4j/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/log4j/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 10:
        # lucene

        f1 = pd.read_csv("../../uncombined_dataset_modified/lucene/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/lucene/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 11:
        # poiV0

        f1 = pd.read_csv("../../uncombined_dataset_modified/poi/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/poi/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 12:
        # poiV1

        f1 = pd.read_csv("../../uncombined_dataset_modified/poi/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/poi/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 13:
        # synapse

        f1 = pd.read_csv("../../uncombined_dataset_modified/synapse/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/synapse/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 14:
        # velocity

        f1 = pd.read_csv("../../uncombined_dataset_modified/velocity/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/velocity/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 15:
        # xercesV0

        f1 = pd.read_csv("../../uncombined_dataset_modified/xerces/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/xerces/2.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 16:
        # xercesV1

        f1 = pd.read_csv("../../uncombined_dataset_modified/xerces/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/xerces/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    return df1,df2


def readDataset_for_testing(datasets):

    if datasets == 0:
        # antV0

        f1 = pd.read_csv("../../uncombined_dataset_modified/ant/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/ant/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 1:
        # antV1

        f1 = pd.read_csv("../../uncombined_dataset_modified/ant/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/ant/4.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 2:
        # antV2

        f1 = pd.read_csv("../../uncombined_dataset_modified/ant/3.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/ant/5.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 3:
        # camelV0

        f1 = pd.read_csv("../../uncombined_dataset_modified/camel/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/camel/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 4:
        # camelV1

        f1 = pd.read_csv("../../uncombined_dataset_modified/camel/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/camel/4.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 5:
        # ivy

        f1 = pd.read_csv("../../uncombined_dataset_modified/ivy/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/ivy/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 6:
        # jeditV0

        f1 = pd.read_csv("../../uncombined_dataset_modified/jedit/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/jedit/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 7:
        # jeditV1

        f1 = pd.read_csv("../../uncombined_dataset_modified/jedit/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/jedit/4.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 8:
        # jeditV2

        f1 = pd.read_csv("../../uncombined_dataset_modified/jedit/3.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/jedit/5.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 9:
        # log4j

        f1 = pd.read_csv("../../uncombined_dataset_modified/log4j/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/log4j/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 10:
        # lucene

        f1 = pd.read_csv("../../uncombined_dataset_modified/lucene/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/lucene/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 11:
        # poiV0

        f1 = pd.read_csv("../../uncombined_dataset_modified/poi/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/poi/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 12:
        # poiV1

        f1 = pd.read_csv("../../uncombined_dataset_modified/poi/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/poi/4.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 13:
        # synapse

        f1 = pd.read_csv("../../uncombined_dataset_modified/synapse/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/synapse/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 14:
        # velocity

        f1 = pd.read_csv("../../uncombined_dataset_modified/velocity/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/velocity/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 15:
        # xercesV0

        f1 = pd.read_csv("../../uncombined_dataset_modified/xerces/1.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/xerces/3.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    elif datasets == 16:
        # xercesV1

        f1 = pd.read_csv("../../uncombined_dataset_modified/xerces/2.csv", delimiter=",")
        df1 = pd.DataFrame(f1)

        f2 = pd.read_csv("../../uncombined_dataset_modified/xerces/4.csv", delimiter=",")
        df2 = pd.DataFrame(f2)

    return df1,df2