conda create -n knn python=3.7
conda activate knn
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install pip
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=knn

python -m pip install detectron2 -f \
https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html


conda install tqdm
conda install pandas
conda install matplotlib
conda install seaborn
conda install scikit-learn 
conda install scipy
conda install -c conda-forge faiss-gpu

conda install -c iopath iopath
conda install simplejson
conda install termcolor
pip install submitit

conda install -c conda-forge ipywidgets