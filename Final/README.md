# Final

This folder contains the work I did to collect information from the [Riot API](https://developer.riotgames.com/), and subsequent models.

I also wrote a report containing my findings that can be found [here](https://nbviewer.jupyter.org/github/kurtisdavid/LSTM_TimeSeriesRegression/blob/master/Final/FinalReport_KurtisDavid.ipynb).

If I were to redo this project again, I would:
- Reframe my data collection to create a general model for specific champion matchups, however, due to the 
constraints of the developer rate limit, that would take too long for a semester project.
- Rewrite everything to use pandas instead (didn't have experience at the time)
- Focus on the multi-output regression, rather than on classifying (to predict future statistics such as CS@20, gold@20, first blood chance)
