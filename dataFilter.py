import os
import shutil
import xml.etree.ElementTree as ET

rawDatasetPath = "./VGG-Face2/data/test/"
parsedDest = "./parsed/"
os.makedirs(parsedDest, exist_ok=True)

VMERXMLPath = "./VMER_dataset/finalTest.xml"

tree = ET.parse(VMERXMLPath)
root = tree.getroot()

def getEthName(eth: int) -> str:
    match eth:
        case 1:
            return "african"
        case 2:
            return "asian"
        case 3:
            return "latin"
        case 4:
            return "indian"
        case _:
            return "unknown"

n = 0
for person in root.findall("subject"):
    print(f"n={n} Id: {person.findtext("id")} Eth: {person.findtext("ethnicity")}")
    ethPath = os.path.join(parsedDest, getEthName(int(person.findtext("ethnicity") or "0") ))
    print(ethPath)
    os.makedirs(ethPath, exist_ok=True)
    originPath = os.path.join(rawDatasetPath, person.findtext("id") or "eeee")
    firstImage = os.listdir(originPath)[0]
    shutil.copy(os.path.join(originPath, firstImage), os.path.join(ethPath, (person.findtext("id") or "") + ".jpg" ))
    
    n += 1
