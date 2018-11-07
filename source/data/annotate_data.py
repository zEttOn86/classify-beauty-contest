# coding:utf-8
import os, time
import random
import requests
from bs4 import BeautifulSoup
from itertools import chain
import re
import urllib.request
import pandas as pd

BASE_URL = 'http://misscolle.com'
RESULT_DIR = '../../data/raw'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, RESULT_DIR)

# データを取得
cols = ['path', 'college', 'year', 'entry_num', 'label']
df = pd.DataFrame(columns=cols)
with open('{}/page_urls.txt'.format(OUTPUT_DIR)) as f:
  for url in f:
    # Make directories for saving images
    college_name, year = re.findall(r'(\d+|\D+)', url.strip().split('/')[-1])
    dirpath = '{}/{}'.format(college_name, year)
    print('Download from: {}'.format(dirpath))
    r = requests.get('{}'.format(url.strip()))
    soup = BeautifulSoup(r.text, 'html.parser')

    info = soup.find_all('div', class_='info')
    #display(info)
    award_infos = map(lambda path: path.find_all('span', class_=re.compile("award*")), info)
    entry_infos = map(lambda path: path.find_all('span', class_=re.compile("entry*")), info)

    for entry_info, award_info in zip(entry_infos, award_infos):

      college = college_name
      year = year
      entry_num = '{:04d}'.format(int(entry_info[0].getText().split(' ')[1]))
      path = '{}/{}'.format(dirpath, entry_num)
      award_info = award_info[0].getText()
      # Annotate
      if re.match(r'グランプリ', award_info):
        label = 0
      elif re.match(r'準グランプリ', award_info):
        label = 1
      else:
        label = 2

      tmp_se = pd.Series( [ path, college, year, entry_num, label ], index=df.columns )
      df = df.append( tmp_se, ignore_index=True )

    # Add random waiting time (4 - 6 sec)
    time.sleep(4 + random.randint(0, 2))

  df.to_csv('{}/annotation.csv'.format(OUTPUT_DIR), index=False, encoding='utf-8', mode='w')
  print(df.head(30))
