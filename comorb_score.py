import pandas as pd
import json
import re
import math
import argparse
import ast
import numpy as np


# Elixhauser index
ELIXHAUSER = 0

CI_MAP = {
            ELIXHAUSER: json.load(open("data/elixhauser.json", "r"))
          }

def _calculate_comorbidity_score(codeList, ccMap, biasTerm=0):
    patCCDict = {"bias": biasTerm}
    for cc, value in ccMap.items():
        for icd9pat in value["pattern"]:
            # compile the pattern
            pat = re.compile(icd9pat)
            patMatch = list(filter(pat.match, codeList))
            if len(patMatch) > 0:
                patCCDict[cc] = value["weight"]
                break # no need to continue with this icd pattern list
    return patCCDict


def calculate_ci(patICDList, version=ELIXHAUSER):
    """
    Calculate comorbidity index
    """
    if version not in CI_MAP:
        raise ValueError("Unsupported comorbidity index")
    patCCMap = _calculate_comorbidity_score(patICDList, CI_MAP[version])
    return sum(patCCMap.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="filename for input file")
    parser.add_argument("outfile", help="filename for the output index")
    parser.add_argument("--icdCol", default="icd9", help="column name for the icd-9 code")
    args = parser.parse_args()

    icdCol = args.icdCol
    # read in the file
    patMeasDF = pd.read_csv(args.infile)
    # make sure icd-9 is icd9
    patMeasDF[icdCol] = patMeasDF[icdCol].apply(lambda x: ast.literal_eval(x))
    # calculate Elixhauser
    patMeasDF['elix'] = patMeasDF[icdCol].apply(calculate_ci, version=ELIXHAUSER)
    patMeasDF.to_csv(args.outfile, index=False)


if __name__ == "__main__":
    main()
