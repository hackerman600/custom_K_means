# custom_K_means

K means clustering algorithim implementation. I chose the centroids randomly, the centroids being random datapoints from the dataset. I added a features where the process of clusering was repeated x amount of times, what followed was calculating the median of absolute differences between the centroids and all datapoints clustering around them. I then averaged the median of absolute differences over the k centroids. This allowed me to go back and choose the centroid initialisation attempt that resulted in the most dense clusters meaning the most similar datapoints being clustered together.    
