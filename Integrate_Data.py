import csv
import xml.etree.cElementTree as ET
def cleanData():

    cleanedData = []
    tree = ET.parse("reviews.xml")
    root = tree.getroot()

    for child in root:
        if(child[0].text == None):
            continue
        newdata = ' '.join(word for word in child[0].text.split(" ") if len(word) < 12)
        cleanedData.append(newdata)


    #write cleaned data
    rootCleaned = ET.Element("root")
    for string in cleanedData:
        doc = ET.SubElement(rootCleaned, "doc")
        data = ET.SubElement(doc, "data")
        sentiment = ET.SubElement(doc, "sentiment")
        data.text = string

    treeCleaned = ET.ElementTree(rootCleaned)
    treeCleaned.write("reviews_cleaned.xml")

cleanData()

