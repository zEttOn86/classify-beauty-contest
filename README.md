# Classify beauty contest

### Purpose

グランプリと準グランプリを取る人を当てる

### Procedures

#### 1. データの用意

1. 画像のデータとアノテーションを以下のサイトよりスクレイビングする

    - [MISS COLLE](https://misscolle.com/)

    ```
    # Scrape image data
    python source/data/scrape_data.py
    # Scrape annotation info
    python source/data/annotate_data.py
    ```

2. `misshoge` と `hoge` のような表記ゆれが存在するので、これを消す

3. 顔のクロップ

4. 使用データの選択


### References

- [ある美女が，どの大学にいそうかを CNN で判別する](https://qiita.com/pika_shi/items/3c8ab1a8ecc655b33851)
