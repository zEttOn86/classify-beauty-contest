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
    if path[-1] == '/':  # Normalize
        path = path[:-1]
    f.write('{}\n'.format(path))

# データを取得
with open('{}/20181102page_urls.txt'.format(OUTPUT_DIR)) as f:
  for url in f:
    # Make directories for saving images
    college_name, year = re.findall(r'(\d+|\D+)', url.strip().split('/')[-1])
    dirpath = '{}/photos/{}/{}'.format(OUTPUT_DIR, college_name, year)
    print('Download from: {}'.format(dirpath))
    r = requests.get('{}/photo'.format(url.strip()))
    soup = BeautifulSoup(r.text, 'html.parser')

    photos = soup.find_all('li', class_='photo')
    paths = map(lambda path: path.find('a').get('href'), photos)

    for path in paths:
      entry_num, filename = path.split('?')[0].split('/')[-2:]
      output_dir = '{}/{:04d}'.format(dirpath, int(entry_num))
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      filepath = '{}/{:04d}.jpg'.format(output_dir, int(os.path.splitext(filename)[0]))
      # Download image file
      try:
        urllib.request.urlretrieve('{}{}'.format(BASE_URL, path), filepath)
      except:
        print('I cant download the images')
      # Add random waiting time (4 - 6 sec)
      time.sleep(4 + random.randint(0, 2))
