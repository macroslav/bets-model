def get_cv_data(data, start_from_season: str = '2017-2018',
                unit: str = 'days',
                days: int = 7,
                weeks: int = 1,
                benchmark_league: str = 'premier-league',
                start_date=None
                ):

    if start_date:
        start_time = data[data.day == start_date].timestamp_date.min()
    else:
        start_time = data[(data.season == start_from_season) & (data.league == benchmark_league)].timestamp_date.min()
    finish_time = data.timestamp_date.max()

    window = 24 * 3600
    if unit == 'days':
        window *= days
    elif unit == 'weeks':
        window *= weeks * 7

    for period in range(start_time, finish_time, window):
        train_cv = data.loc[data.timestamp_date < period]
        val_cv = data.loc[(data.timestamp_date > period) & (data.timestamp_date <= period + window)]
        yield train_cv, val_cv
