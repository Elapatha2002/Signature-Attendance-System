import xml.etree.ElementTree as ET
import os

def load_course_info(xml_path):

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"The file {xml_path} is missing.")

    tree = ET.parse(xml_path)
    root = tree.getroot()


    pass