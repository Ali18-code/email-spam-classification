"""
Downloads the SMS Spam Collection dataset (the same labelled ham/spam dataset
distributed on Kaggle as "SMS Spam Collection Dataset") and saves it as
sms_spam.tsv in the current directory.

Alternative: download manually from Kaggle
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
and place the file as `sms_spam.tsv` (tab-separated, columns: label, message)
in this folder.
"""
import urllib.request

URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
OUT_FILE = "sms_spam.tsv"

if __name__ == "__main__":
    print(f"Downloading dataset from {URL} ...")
    urllib.request.urlretrieve(URL, OUT_FILE)
    print(f"Saved to {OUT_FILE}")
