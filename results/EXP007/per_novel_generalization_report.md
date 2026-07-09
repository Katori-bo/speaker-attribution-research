# EXP007: Per-Novel Generalization Report

Does the Top 3 feature representation suffice for all novels, or do complex novels demand richer context?

![Cast vs Features](/home/Aditya/speaker-attribution-research/results/EXP007/novel_cast_vs_features.png)

| Novel                        |   Accuracy_Top3 |   Accuracy_All |   d_Accuracy |   Num_Speakers |   Avg_Candidate_Set_Size |   Avg_Quote_Length |
|:-----------------------------|----------------:|---------------:|-------------:|---------------:|-------------------------:|-------------------:|
| AlicesAdventuresInWonderland |        0.863142 |       0.880249 |   0.0171073  |             25 |                  5.37644 |            63.0091 |
| PrideAndPrejudice            |        0.790657 |       0.798443 |   0.00778547 |             22 |                  5.8126  |           215.445  |
| SenseAndSensibility          |        0.785075 |       0.79204  |   0.00696517 |             19 |                  4.95576 |           265.892  |
| MansfieldPark                |        0.721951 |       0.72878  |   0.00682927 |             20 |                  6.09594 |           276.172  |
| TheSportOfTheGods            |        0.761404 |       0.764912 |   0.00350877 |             27 |                  4.65156 |           114.814  |
| Emma                         |        0.804895 |       0.805574 |   0.00067981 |             15 |                  4.76208 |           264.994  |
| TheGambler                   |        0.803419 |       0.80057  |  -0.002849   |             11 |                  4.45241 |           168.382  |
| Persuasion                   |        0.738426 |       0.731481 |  -0.00694444 |             22 |                  6.04098 |           288.946  |
| OliverTwist                  |        0.853952 |       0.845361 |  -0.00859107 |             66 |                  4.61162 |           111.547  |
| WinnieThePooh                |        0.820513 |       0.807082 |  -0.013431   |             12 |                  4.04778 |            55.7867 |
| AHandfulOfDust               |        0.825861 |       0.81076  |  -0.0151015  |             55 |                  4.17886 |            73.5957 |
| TheSignOfTheFour             |        0.857143 |       0.837438 |  -0.0197044  |             16 |                  4.09687 |           280.754  |
| NorthangerAbbey              |        0.787995 |       0.766284 |  -0.0217114  |             13 |                  4.13777 |           179.03   |
| ARoomWithAView               |        0.832355 |       0.810176 |  -0.0221787  |             22 |                  4.09486 |           100.064  |
| ThePictureOfDorianGray       |        0.922321 |       0.9      |  -0.0223214  |             24 |                  3.30316 |           170.208  |
| TheMysteriousAffairAtStyles  |        0.857061 |       0.832951 |  -0.0241102  |             26 |                  5.46212 |           109.159  |
| HowardsEnd                   |        0.877642 |       0.852846 |  -0.0247967  |             36 |                  3.76362 |           101.283  |
| TheAwakening                 |        0.865065 |       0.837338 |  -0.0277264  |             13 |                  3.54452 |           101.78   |
| TheAgeOfInnocence            |        0.781734 |       0.75387  |  -0.0278638  |             27 |                  3.95334 |           107.209  |
| TheInvisibleMan              |        0.913915 |       0.883255 |  -0.0306604  |             22 |                  3.2865  |           105.515  |
| APassageToIndia              |        0.821741 |       0.789799 |  -0.0319423  |             39 |                  4.8521  |            89.6938 |
| HardTimes                    |        0.84713  |       0.815087 |  -0.0320427  |             23 |                  4.1448  |           163.373  |
| DaisyMiller                  |        0.925287 |       0.89272  |  -0.032567   |              9 |                  3.71273 |            84.478  |
| TheManWhoWasThursday         |        0.835781 |       0.801068 |  -0.034713   |             21 |                  4.85857 |           157.209  |
| TheSunAlsoRises              |        0.893891 |       0.856109 |  -0.0377814  |             37 |                  3.51226 |            48.8627 |
| WhereAngelsFearToTread       |        0.843158 |       0.803158 |  -0.04       |             14 |                  3.398   |            99.9373 |
| AnneOfGreenGables            |        0.817774 |       0.771095 |  -0.0466786  |             21 |                  4.18478 |           320.291  |
| NightAndDay                  |        0.79879  |       0.741301 |  -0.0574887  |             30 |                  3.59063 |            96.0729 |