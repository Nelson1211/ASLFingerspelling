# ASL Finger Spelling

This project focuses on converting ASL alphabets from videos into English characters and displaying them. It also allows the user to use a video showing a word without same consecutive alphabets.

# Prerequisites

1. Tensorflow v1.4
2. Sklearn
3. OpenCV
4. Numpy

# Running the code

1. Name all the alphabet videos as alphabet.mp4 format eg. A.mp4 for the alphabet A.
2. Place all the videos to be recognized in the data folder.
3. Run the following command to predict the alphabets and get the F1 score.
```
python3 recognize_alphabet.py
```
4. Run the following command to predict words.
```
python3 alphabet_mode_main.py
```

# Acknowledgements

This project has been possible because of the [https://github.com/victordibia/handtracking](https://github.com/victordibia/handtracking) repository as it eased the work of palm detection which is the foundation stone of this project. The second major contribution in this project came from [Professor Ayan Banerjee](https://isearch.asu.edu/profile/1014358) who provided the model trained on ASL alphabets dataset from [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet).