import json
import itertools
import argparse
import collections
import re

def _abbr_filter(abbr):
    if len(abbr) < 2:
        return False
    return True


def _clean_abbr(abbrList):
    # append the or to make life a bit easier
    abbr = (" or ".join(abbrList)).strip()
    if abbr is None or abbr == "":
        return None
    # split based on comma or ;
    tmpAbbr = re.split(",|;", abbr)
    # then resplit based on or
    tmpAbbr = list(map(lambda x: x.split(" or "), tmpAbbr))
    # flatten the list
    tmpAbbr = itertools.chain(*tmpAbbr)
    # clean the list to remove spaces
    tmpAbbr = list(map(lambda x: x.strip(), tmpAbbr))
    # clean the list with some filtering
    tmpAbbr = list(filter(_abbr_filter, tmpAbbr))
    return tmpAbbr


def _clean_qual_name(fullList):
    full = "".join(fullList)
    # clean up the parenthsis
    full = re.sub(r'\([^)]*\)', '', full)
    # full = re.sub(r'\ from Latin.+', '', full, flags=re.IGNORECASE)
    # full = re.sub(r'\ from the Latin.+', '', full, flags=re.IGNORECASE)
    # full = re.sub(r'\ from Middle English.+', '', full, flags=re.IGNORECASE)
    return full



def parse_json(abbrDict):
    finalAbbr = collections.defaultdict(list)
    # setup the dictionary with the final abbreviations
    for nl in abbrDict:
        abbrList = _clean_abbr(nl['abbr'])
        if abbrList is None:
            continue
        qualName = _clean_qual_name(nl['full'])
        if " or " in qualName:
            continue
        for ab in abbrList:
            finalAbbr[ab.strip()].append(qualName)
    return finalAbbr


def deleteMultiKeys(abbrDict):
    finalAbbr = {}
    for k, v in abbrDict.items():
        # map the values to lowercase set
        syn = set(map(lambda x: x.lower(), v))
        if len(syn) == 1:
            finalAbbr[k] = v[0].lower()
    return finalAbbr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outputFile", help="filename for updated abbreviations")
    parser.add_argument("-i", help="input json files to parse", nargs='+', required=True)
    args = parser.parse_args()

    jsonAbbrFiles = args.i
    finalAbbrDict = {}
    for jsonFile in jsonAbbrFiles:
        print("Loading and cleaning file:" + jsonFile)
        # open the json abbreviation file
        jsonAbbr = parse_json(json.load(open(jsonFile, 'r')))
        jsonClenedAbbr = deleteMultiKeys(jsonAbbr)
        finalAbbrDict.update(jsonClenedAbbr)
    with open(args.outputFile, 'w') as outfile:
        json.dump(finalAbbrDict, outfile, indent=2)


if __name__ == "__main__":
    main()