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

    ```
    python source\data\change_orthographic_variants.py
    ```

3. 顔のクロップ

    - 分類器は，[ここ](https://github.com/opencv/opencv/tree/master/data/haarcascades)から `data/external` にダウンロード

    - もしくは，[ここ](https://github.com/opencv/opencv_contrib/tree/master/modules/face/data/cascades)

    ```
    python source\data\face_detecter.py
    ```

    - 次のように検出に失敗するので， `face_recognition` を使用する

      <img src='report/fig1.png'>

    - How to install

      ```
      # Make sure already installed visual studio 2015
      pip install cmake
      pip install face_recognition
      # Ref. https://github.com/ageitgey/face_recognition/issues/175#issuecomment-426216930
      ```

    



4. 使用データの選択


### References

- [ある美女が，どの大学にいそうかを CNN で判別する](https://qiita.com/pika_shi/items/3c8ab1a8ecc655b33851)
