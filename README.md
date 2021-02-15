# GenreByLyrics

Adapted https://riazhedayati.github.io/blog/predict-song-genre-pt1/ for the genrebycontent file to quickly get the top songs.

This used only top 100 songs of each genre for a total of 500 songs. 
For the neural network 500 songs is too little so np.repeat() was used to duplicate entries.

When np.repeat was used with large numbers it could overfit to the training.

With the given dataset and changing the hyperparameters arbitrarily the best accuracy was
5/17 or ~30%
