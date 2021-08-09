import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree

dir = 'C:\Users\hje06\OneDrive\문서\AI school\VehicleDetection\Vehicle\test\annotations'

# parse xml file
doc = ET.parse(file_name)
# get root node
root = doc.getroot()

