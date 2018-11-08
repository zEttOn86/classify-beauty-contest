import os, time
import random
import requests
from bs4 import BeautifulSoup
from itertools import chain
import re
import urllib.request

BASE_URL = 'http://misscolle.com'
RESULT_DIR = '../../data/raw'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, RESULT_DIR)

# URLを取得
r = requests.get('{}/versions'.format(BASE_URL))
soup = BeautifulSoup(r.text, 'html.parser')

columns = soup.find_all('ul', class_='columns')
#display(columns)
atags = map(lambda column: column.find_all('a'), columns)

with open('{}/page_urls.txt'.format(OUTPUT_DIR), 'w') as f:
    for _ in chain.from_iterable(atags):
        path = _.get('href')
        if not path.startswith('http'):  # Relative path
            path = '{}{}'.format(BASE_URL, path)
        else:
            path = '{}/{}'.format(BASE_URL, path.split('/')[-2].replace('miss', ''))
        if path[-1] == '/':  # Normalize
            path = path[:-1]
        f.write('{}\n'.format(path))

# データを取得
with open('{}/page_urls.txt'.format(OUTPUT_DIR)) as f:
    for url in f:
        # Make directories for saving images
        college_name, year = re.findall(r'(\d+|\D+)', url.strip().split('/')[-1])
        dirpath = '{}/photos/{}/{}'.format(OUTPUT_DIR, college_name, year)
        print('Download from: {}'.format(url.strip()))
        r = requests.get('{}/photo'.format(url.strip()))
        soup = BeautifulSoup(r.text, 'html.parser')

        photos = soup.find_all('li', class_='photo')
        info = soup.find_all('li', class_='photo_entry')

        paths = map(lambda path: path.find('a').get('href'), photos)
        entry_infos = list(map(lambda path: path.find_all('span', class_=re.compile("entry*")), info))
        entry_num_counter = 1
        tmp_entry_num = -1000
        for path in paths:
            tmp, filename = path.split('?')[0].split('/')[-2:]
            tmp = int(tmp)
            if tmp > tmp_entry_num:
                tmp_entry_num = tmp
                entry_num = int(entry_infos[entry_num_counter-1][0].getText().split(' ')[1])
                entry_num_counter += 1
            output_dir = '{}/{:04d}'.format(dirpath, entry_num)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = '{}/{:04d}.jpg'.format(output_dir, int(os.path.splitext(filename)[0]))
            # Download image file
            try:
                urllib.request.urlretrieve('{}{}'.format(BASE_URL, path), filepath)
            except:
                print('I cant download the images')
            # Add random waiting time (1 - 3 sec)
            time.sleep(1 + random.randint(1, 2))
