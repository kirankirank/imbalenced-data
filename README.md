# imbalenced-data


I.	Under sampling( if the data 80 and 20. So we take randomly 20% data in 80 of data)
        A.	Random:
         it is a fast and easy way to balance the data by randomly selecting a subset of data for the targeted classes.
        B.	Centroid:
        removing samples which do not agree  with their neighborhood . For each sample in the class to be under-sampled, the nearest-neighbors are computed and if the selection criterion is not fulfilled, the sample is removed.
        C.	Cluster centroids based: 
        Cluster Centroids makes use of K-means to reduce the number of samples. Therefore, each class will be synthesized with the centroids of the K-means method instead of the original samples.
        D.	Near Miss
        E.	TomkLinks
II.	Over sampling
        A.	Random Over Sampler:
        to generate new samples by randomly sampling with replacement the current available samples.)(this one duplicating values increasing means generating same data)
        B.	SMOTE:
        this one in use to create new data
III.	Combo of (under sampling + over sampling)
        A.	SMOTEENN
IV.	Ensemble (this one for ensemble)
V.	Batch approach (applicable to  tensor flow and keras)

