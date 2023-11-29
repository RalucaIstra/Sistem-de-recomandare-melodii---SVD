# Sistem-de-recomandare-melodii---SVD

This repository contains the code for a music recommendation system based on collaborative filtering using Singular Value Decomposition (SVD). The recommendation system analyzes user listening patterns and suggests songs based on user preferences.

Overview

The recommendation system is implemented in Python using the NumPy, Pandas, SciPy, and Matplotlib libraries. It includes the following components:

    Data Cleaning: The initial dataset is cleaned by removing users and songs with minimal replays to improve the accuracy of recommendations.
    Singular Value Decomposition (SVD): The system uses SVD to decompose the user-song interaction matrix into three matrices: U, sigma, and Vt. This helps in understanding latent features and capturing patterns in the data.
    Efficiency Evaluation: The system evaluates its efficiency by calculating the Mean Squared Error (MSE) between the original and reconstructed matrices.
    Music Recommendations: The system recommends songs to users based on their listening history and preferences.

File Structure

    colocviu.py: Contains the main code for data cleaning, SVD, efficiency evaluation, and music recommendations.
    user.csv: Dataset containing user listening history.
    music(1).csv: Dataset containing information about songs.
    user_curatat1.csv: Cleaned user dataset after filtering out minimal replays.
    README.md: Provides an overview of the project and instructions.

Contributing
Contributions are welcome! Feel free to submit issues or pull requests.
