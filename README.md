## Installation
Clone the repository: 
```bash
git clone --recurse-submodules git@github.com:AyonRRahman/Thesis.git
```
## Download the Eiffel Tower dataset
```bash
cd data
chmod +x datasetup.sh
./datasetup.sh
```

## Download the Kitty Raw dataset
Official link to download the dataset:
http://www.cvlibs.net/download.php?file=raw_data_downloader.zip
```bash
cd data
mkdir Kitty_raw
cp Kitty_raw_data_downloader.sh Kitty_raw
cd Kitty_raw
./Kitty_raw_data_downloader.sh

```

