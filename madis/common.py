"""Common utility functions for MADIS processing."""

from datetime import datetime

from dateutil import rrule


def get_dates(start_date, end_date):
    """Get a list of dates between start_date and end_date.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    list
        List of datetime objects for each hour in the date range.
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    dates = list(rrule.rrule(rrule.HOURLY, dtstart=start, until=end))[:-1]
    return dates
