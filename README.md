# FaceRotation

- DemoFaceAlignment.py
  - 動画からランドマークの検出
  - python DemoFaceAlignment.py <path/to/動画ファイル> <3dを指定すると3次元のランドマークを抽出>
  - 出力は detected(_3d)/ フォルダ下に動画とランドマークのnumpyファイル
  - set_intervalをいじると何フレーム毎に処理を行うかを指定できる(デフォルトは5フレーム毎)
  
- landmark_detection.py
  - DemoFaceAlignment.pyの動画に複数人が入ってる場合ターゲットを一人に絞る(無理矢理絞り込んでるので使わない方がいいです)
  - python landmark_detection.py <path/to/landmark> <開始時間>
  - 動画の横幅MIN_AREA〜MAX_AREAの中に入っている人物を追跡
  
- rotation_lmark.py
  - DemoFaceAlignment.pyで出力したランドマークのnumpyファイルを正面化する
  - python rotation_lmark.py <path/to/ランドマークファイル>
  - 出力は rotated/動画名/npy/ フォルダ下にnumpyファイル

- FaceNormalization.py
  - DemoFaceAlignment.pyの動画とランドマークを使って顔の正面化+目と口の抽出
  - python FaceNormalization.py <path/to/動画ファイル> <path/to/ランドマーク>
  - 出力は rotated/動画名/leye/ 等のフォルダ下に画像、rotated/動画名/ 下に動画

- IMGtoFrontalLandmark.py
  - 画像を入力としてランドマークを正面化
  - python IMGtoFrontalLandmark.py <path/to/images/> <path/to/output/>
  - 出力はoutput下にランドマーク
  
- IMGtoeyes_mouth.py
  - 画像を入力として顔画像を正面化+目と口の抽出
  - python IMGtoeyes_mouth.py <path/to/dataset>
  - 出力は path/to/dataset/data/ 下の顔画像に対し， path/to/dataset/leye/ 等の下に画像を出力
