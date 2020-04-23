# Created by: c00k1ez (https://github.com/c00k1ez)

import urllib.request
import zipfile
import os

DATA_LINK = 'http://www.manythings.org/anki/rus-eng.zip'

DIR = './data/'

FILE_PATH = DIR + 'rus-eng.zip'

class AppURLopener(urllib.request.FancyURLopener):
        version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"
    

if __name__ == '__main__':

    if not os.path.isdir(DIR):
        os.mkdir(DIR)

    urllib._urlopener = AppURLopener()

    urllib._urlopener.retrieve(DATA_LINK, FILE_PATH)

    with zipfile.ZipFile(FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(DIR)

    os.remove(FILE_PATH)