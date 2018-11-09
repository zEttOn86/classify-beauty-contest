### Dataset organization

以下に各フォルダの役割を示す．

```
├─external          <--- 外部から取得したモデルなど
├─interim           <--- rawから表記ゆれを直した結果
│  ├─abeno
│  ├─aichi
│  ├─aitama
│  ├─ ...
│  └─ynu
├─preprocessed       <--- face_recognitionによる顔検出し，224x224にリサイズした結果
│  ├─abeno
│  ├─aichi
│  ├─aitama
│  ├─ ...
│  └─ynu
├─preprocessed_opencv <--- OpenCVによる顔検出の結果
│  └─abeno
└─raw                 <--- スクレイビングの生データ
```

画像は以下のように格納されている

```
├─大学名 (i.e. ynu)
    ├─年 (i.e. 2017)
        ├─コンテスト参加者番号 (i.e. 0001)
            ├─画像番号  (i.e. 0001.jpg)
```

annotation.csvの内訳は以下のようになっている

```
各ラベルの意味

- 0: グランプリ

- 1: 準グランプリ

- 2: その他
```

| || | | |
| :---: |:---: | :---: | :---: | :---: |
| path | 大学名 |年 | 症例番号 | ラベル |
