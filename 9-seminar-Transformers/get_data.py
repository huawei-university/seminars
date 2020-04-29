# Created by: c00k1ez (https://github.com/c00k1ez)

import urllib.request
import zipfile
import requests
import os
import zipfile


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

DATA_LINK = 'https://raw.githubusercontent.com/Phylliida/Dialogue-Datasets/master/TwitterLowerAsciiCorpus.txt'

DIR = './data/'

FILE_PATH = DIR + 'TwitterLowerAsciiCorpus.txt'

class AppURLopener(urllib.request.FancyURLopener):
        version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"
    

if __name__ == '__main__':

    if not os.path.isdir(DIR):
        os.mkdir(DIR)

    urllib._urlopener = AppURLopener()

    urllib._urlopener.retrieve(DATA_LINK, FILE_PATH)

    file_id = '1m3QXJ3tAU0CIf13n17e9y61EKFoCIeof'
    destination = DIR + 'question_pairs.zip'
    download_file_from_google_drive(file_id, destination)

    with zipfile.ZipFile(DIR + 'question_pairs.zip', 'r') as zip_ref:
        zip_ref.extractall(DIR)
    
    os.remove(DIR + 'question_pairs.zip')
