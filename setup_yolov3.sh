make
wget https://pjreddie.com/media/files/darknet53.conv.74
wget https://raw.githubusercontent.com/YHRen/data_utilities/master/id2name.txt
wget https://raw.githubusercontent.com/YHRen/data_utilities/master/setup_train_xview.py
mv setup_train_xview.py ./scripts_ray/
yes | apt install python3 python3-pip
pip3 install numpy matplotlib tqdm pillow
python3 scripts_ray/setup_train_xview.py -i /data/xView_train.geojson -m /data/train_images -d id2name.txt -o ./chip_data/ -r 672
find $(pwd)/chip_data/chip_train/ -iname '*.jpg' >./meta/train_img.txt
find $(pwd)/chip_data/chip_valid/ -iname '*.jpg' >./meta/valid_img.txt
mkdir -p meta/checkpoint
echo -e "classes = 62\ntrain = $(pwd)/meta/train_img.txt\ntest = $(pwd)/meta/valid_img.txt\nnames = $(pwd)/meta/xview.names\nbackup = $(pwd)/meta/checkpoint" >./meta/xview.data
