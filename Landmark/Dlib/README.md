- Dilbライブラリのダウンロード
    
    Pythonのパッケージインデックス([PyPI](https://pypi.org))からインストール
    
    pip install dlib
    
    anaconda環境の場合、ターミナルで次のコマンドを実行し仮想環境にDlibライブラリを入れることもできる
    
    conda install -c conda-forge dlib
    
- モデルのダウンロード >> modelディレクトリに格納
    - 顔検出器
        
        [Dlib公式サイト](http://dlib.net/files/)で「mmod_human_face_detector.dat.bz2」を直接ダウンロードするかターミナルで次のコマンドを実行
        
        curl -O http://dlib.net/files/mmod_human_face_detector.dat.bz2
        bzip2 -d mmod_human_face_detector.dat.bz2
        
    - ランドマーク検出器
        
        [Dlib公式サイト](http://dlib.net/files/)で「shape_predictor_68_face_landmarks.dat.bz2」を直接ダウンロードするかターミナルで次のコマンドを実行
        
        curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        bzip2 -d shape_predictor_68_face_landmarks.dat.bz2