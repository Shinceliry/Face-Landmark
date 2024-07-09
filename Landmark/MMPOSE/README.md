- Pytorchのインストール
Pytorchのバージョンが新すぎるとmmposeが対応してないために以降のインストールがうまくいかない。そのため新しく仮想環境を作成し, Pytorchのバージョンを指定してインストールする
    
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
    
- mmengineのインストール
    
    pip install -U openmim
    mim install mmengine
    
- mmcvのインストール
    
    mim install "mmcv>=2.0.1"
    
- mmdetのインストール
    
    mim install "mmdet>=3.1.0"
    
- Xtcocoapi のインストール
    - mmposeライブラリをインストールしようとするとXtcocoapi のインストールできないというエラーが出るのでXtcocoapi を手動でインストールしてからmmposeをインストールする
    - [公式Github](https://github.com/jin-s13/xtcocoapi#extended-coco-api-xtcocotools)をgit cloneする(pipだとうまくいかない)
    
    git clone https://github.com/jin-s13/xtcocoapi.git
    
    - インストールに必要なライブラリをインストールしたのちにXtcocoapiをセットアップ
    
    cd xtcocoapi && pip install -r requirements.txt
    python setup.py install
    
- mmposeライブラリのインストール
    
    mim install mmpose
    
- mmposeをgit cloneする
    - このままではモデルのパス先にモデルがなく, コードが動かない
    - インストールしたらモデルのパスを書き換える
    - gitするフォルダ名がmmposeなのでmmposeというフォルダが別にあるとコンフリクトしてしまうので注意
    
    git clone https://github.com/open-mmlab/mmpose
    
- ffmpegをインストール
    
    conda install ffmpeg