# Speaker Recognition using Deep Learning

## Steps to run the scripts:
1. Fetch and unzip the LibriSpeech clean-100 data.
```
sh fetch.sh
```
2. Extract Mel-Filterbank features for all samples and save to disk. We don't do this during model training as it considerably slows down the training process.
```
python3 data_utils.py
```
3. Run the training algorithm.
```
python3 main.py
```
4. Run train_id tasks or test_id tasks by changing the `CKPT_PATH` in `config.py`, then running either of these:
```
python3 train_id_task.py
```
```
python3 test_id_task.py
```