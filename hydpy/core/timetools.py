# -*- coding: utf-8 -*-
"""This module specifies the handling of dates and periods in *HydPy* projects.

.. _`Time Coordinate`: http://cfconventions.org/Data/cf-conventions/\
cf-conventions-1.7/cf-conventions.html#time-coordinate
"""
# import...
# ...from standard library
import calendar
import collections
import copy
import datetime as datetime_
import numbers
import time
from typing import *
# ...from third party packages
import numpy
# ...from HydPy
import hydpy
from hydpy.core import objecttools

# The import of `_strptime` is not thread save.  The following call of
# `strptime` is supposed to prevent possible problems arising from this bug.
time.strptime('1999', '%Y')


DateConstrArg = Union['Date', datetime_.datetime, str]
PeriodConstrArg = Union['Period', datetime_.timedelta, str]


class Date:
    """Handles a single date.

    We built class the |Date| on top of the Python module |datetime|.
    In essence, it wraps |datetime.datetime| objects and specialise
    this general class on the needs of *HydPy* users.

    |Date| objects can be initialised via |datetime.datetime| objects
    directly:

    >>> import datetime
    >>> date = datetime.datetime(1996, 11, 1, 0, 0, 0)
    >>> from hydpy import Date
    >>> Date(date)
    Date('1996-11-01 00:00:00')

    |Date| objects do not store time zone information.  The |Date| object
    prepared above refers to zero o'clock in the time zone defined by
    |Options.utcoffset| (UTC+01:00 by default).  When the initialisation
    argument provides other time zone information, its date information
    is adjusted, which we show in the following examples, where the
    prepared |datetime.datetime| objects refer to UTC 00:00 and UTC-01:00:

    >>> date = datetime.datetime(1996, 11, 1, 0, 0, 0,
    ...     tzinfo=datetime.timezone(datetime.timedelta(0)))
    >>> Date(date)
    Date('1996-11-01 01:00:00')
    >>> date = datetime.datetime(1996, 11, 1, 0, 0, 0,
    ...     tzinfo=datetime.timezone(datetime.timedelta(hours=-1)))
    >>> Date(date)
    Date('1996-11-01 02:00:00')

    One can change |Options.utcoffset|, but this does not affect already
    existing |Date| objects:

    >>> from hydpy import pub
    >>> pub.options.utcoffset = 0
    >>> temp = Date(date)
    >>> temp
    Date('1996-11-01 01:00:00')
    >>> pub.options.utcoffset = 60
    >>> temp
    Date('1996-11-01 01:00:00')

    Class |Date| accepts |str| objects as alternative constructor arguments.
    These are often more rapidly defined and allow to set the |Date.style|
    property by the way (see the documentation on method |Date.from_string|
    for more examples):

    >>> Date('1996-11-01')
    Date('1996-11-01 00:00:00')
    >>> Date('1996.11.01')
    Date('1996.11.01 00:00:00')

    Invalid arguments types result in the following error:

    >>> Date(1)
    Traceback (most recent call last):
    ...
    TypeError: While trying to initialise a `Date` object based on \
argument `1`, the following error occurred: The supplied argument must \
be either an instance of `Date`, `datetime.datetime`, or `str`.  \
The given arguments type is `int`.

    In contrast to class |datetime.datetime|, class |Date| is mutable:

    >>> date = Date('1996-11-01')
    >>> date.hour = 12
    >>> date
    Date('1996-11-01 12:00:00')

    Unplausible values assigned to property |Date.hour| and its related
    properties result in error messages like the following:

    >>> date.hour = 24
    Traceback (most recent call last):
    ...
    ValueError: While trying to change the hour of the current Date object, \
the following error occurred: hour must be in 0..23

    You can do some math with |Date| objects.  First, you can add |Period|
    objects to shift the date:

    >>> date = Date('2000.01.01')
    >>> date + '1d'
    Date('2000.01.02 00:00:00')
    >>> date += '12h'
    >>> date
    Date('2000.01.01 12:00:00')

    Second, you can subtract both |Period| and other |Date| objects to
    shift the date or determine the time delta, respectively:

    >>> date - '1s'
    Date('2000.01.01 11:59:59')
    >>> date -= '12h'
    >>> date
    Date('2000.01.01 00:00:00')
    >>> date - '2000-01-05'
    Period('-4d')
    >>> '2000.01.01 00:00:30' - date
    Period('30s')

    To try to subtract objects neither interpretable as a |Date|
    nor |Period| object results in the following error:

    >>> date - '1'
    Traceback (most recent call last):
    ...
    TypeError: Object `1` of type `str` cannot be substracted from a \
`Date` instance.

    The comparison operators work as expected:

    >>> d1, d2 = Date('2000-1-1'), Date('2001-1-1')
    >>> d1 < d2, d1 < '2000-1-1', '2001-1-2' < d1
    (True, False, False)
    >>> d1 <= d2, d1 <= '2000-1-1', '2001-1-2' <= d1
    (True, True, False)
    >>> d1 == d2, d1 == '2000-1-1', '2001-1-2' == d1, d1 == '1d'
    (False, True, False, False)
    >>> d1 != d2, d1 != '2000-1-1', '2001-1-2' != d1, d1 != '1d'
    (True, False, True, True)
    >>> d1 >= d2, d1 >= '2000-1-1', '2001-1-2' >= d1
    (False, True, True)
    >>> d1 > d2, d1 > '2000-1-1', '2001-1-2' > d1
    (False, False, True)
    """

    # These are the so far accepted date format strings.
    formatstrings = collections.OrderedDict([
        ('os', '%Y_%m_%d_%H_%M_%S'),
        ('iso2', '%Y-%m-%d %H:%M:%S'),
        ('iso1', '%Y-%m-%dT%H:%M:%S'),
        ('din1', '%d.%m.%Y %H:%M:%S'),
        ('din2', '%Y.%m.%d %H:%M:%S')])
    # The first month of the hydrological year (e.g. November in Germany)
    _firstmonth_wateryear = 11
    _lastformatstring = 'os', formatstrings['os']

    datetime: datetime_.datetime

    def __new__(cls, date: DateConstrArg) -> 'Date':
        try:
            if isinstance(date, Date):
                return cls.from_date(date)
            if isinstance(date, datetime_.datetime):
                return cls.from_datetime(date)
            if isinstance(date, str):
                return cls.from_string(date)
            raise TypeError(
                f'The supplied argument must be either an instance of '
                f'`Date`, `datetime.datetime`, or `str`.  The given '
                f'arguments type is `{objecttools.classname(date)}`.')
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to initialise a `Date` '
                f'object based on argument `{date}`')

    @classmethod
    def from_date(cls, date: 'Date') -> 'Date':
        """Create a new |Date| object based on another |Date| object and
        return it.

        Initialisation from other |Date| objects preserves their
        |Date.style| information:

        >>> from hydpy import Date
        >>> date1 = Date('2000.01.01')
        >>> date2 = Date(date1)
        >>> date1.style = 'iso2'
        >>> date3 = Date.from_date(date1)
        >>> date2
        Date('2000.01.01 00:00:00')
        >>> date3
        Date('2000-01-01 00:00:00')
        """
        self = super().__new__(cls)
        self.datetime = date.datetime
        self.style = getattr(date, 'style', None)
        return self

    @classmethod
    def from_datetime(cls, date: datetime_.datetime) -> 'Date':
        """Create a new |Date| object based on a |datetime.datetime| object
        and return it.

        Initialisation from |datetime.datetime| does not modify the
        default |Date.style| information:

        >>> from hydpy import Date
        >>> from datetime import datetime, timedelta, timezone
        >>> Date.from_datetime(datetime(2000, 1, 1))
        Date('2000-01-01 00:00:00')

        Be aware of the different minimum time resolution of class
        |datetime.datetime| (microseconds) and class |Date| (seconds):

        >>> Date.from_datetime(datetime(2000, 1, 1, microsecond=2))
        Traceback (most recent call last):
        ...
        ValueError: For `Date` instances, the microsecond must be zero, \
but for the given `datetime` object it is `2` instead.

        Due to a different kind of handling time zone information,
        the time zone awareness of |datetime.datetime| objects is
        removed (see the main documentation on class |Date| for
        further information:

        >>> date = Date.from_datetime(
        ...     datetime(2000, 11, 1, tzinfo=timezone(timedelta(0))))
        >>> date
        Date('2000-11-01 01:00:00')
        >>> date.datetime
        datetime.datetime(2000, 11, 1, 1, 0)
        """
        if date.microsecond != 0:
            raise ValueError(
                f'For `Date` instances, the microsecond must be zero, '
                f'but for the given `datetime` object it is '
                f'`{date.microsecond:d}` instead.')
        self = super().__new__(cls)
        utcoffset = date.utcoffset()
        if utcoffset is not None:
            date = (date.replace(tzinfo=None) - utcoffset +
                    datetime_.timedelta(minutes=hydpy.pub.options.utcoffset))
        self.datetime = date
        return self

    @classmethod
    def from_string(cls, date: str) -> 'Date':
        """Create a new |Date| object based on a |datetime.datetime| object
        and return it.

        The given string needs to match one of the following |Date.style|
        patterns.

        The `os` style is applied in text files and folder names and does
        not include any empty spaces or colons:

        >>> Date.from_string('1997_11_01_00_00_00').style
        'os'

        The `iso` styles are more legible and come in two flavours.
        `iso1` following ISO 8601, and `iso2` (which is the default
        style) omits the `T` between date and time:

        >>> Date.from_string('1997-11-01T00:00:00').style
        'iso1'
        >>> Date.from_string('1997-11-01 00:00:00').style
        'iso2'

        The `din` styles rely on points instead of hyphens.  The difference
        between the available flavours lies in the order of the date literals
        (DIN refers to a German norm):

        >>> Date('01.11.1997 00:00:00').style
        'din1'
        >>> Date('1997.11.01 00:00:00').style
        'din2'

        You are allowed to abbreviate the input strings:

        >>> for string in ('1996-11-01 00:00:00',
        ...                '1996-11-01 00:00',
        ...                '1996-11-01 00',
        ...                '1996-11-01'):
        ...     print(Date.from_string(string))
        1996-11-01 00:00:00
        1996-11-01 00:00:00
        1996-11-01 00:00:00
        1996-11-01 00:00:00

        You can combine all styles with ISO time zone identifiers:

        >>> Date.from_string('1997-11-01T00:00:00Z')
        Date('1997-11-01T01:00:00')
        >>> Date.from_string('1997-11-01 00:00:00-11:00')
        Date('1997-11-01 12:00:00')
        >>> Date.from_string('1997-11-01 +13')
        Date('1997-10-31 12:00:00')
        >>> Date.from_string('1997-11-01 +1330')
        Date('1997-10-31 11:30:00')
        >>> Date.from_string('01.11.1997 00-500')
        Date('01.11.1997 06:00:00')

        Poorly formatted date strings result in the following or comparable
        error messages:

        >>> Date.from_string('1997/11/01')
        Traceback (most recent call last):
        ...
        ValueError: The given string `1997/11/01` does not agree with any \
of the supported format styles.

        >>> Date.from_string('1997-11-01 +0000001')
        Traceback (most recent call last):
        ...
        ValueError: While trying to apply the time zone offset defined \
by string `1997-11-01 00:00:00`, the following error occurred: \
wrong number of offset characters

        >>> Date.from_string('1997-11-01 +0X:00')
        Traceback (most recent call last):
        ...
        ValueError: While trying to apply the time zone offset defined \
by string `1997-11-01 00:00:00`, the following error occurred: \
invalid literal for int() with base 10: '0X'
        """
        # pylint: disable=protected-access
        # due to pylint issue https://github.com/PyCQA/pylint/issues/1159
        self = super().__new__(cls)
        substring, offset = self._extract_offset(date)
        vars(self)['style'], date = self._extract_date(substring, date)
        self.datetime = self._modify_date(date, offset, date)
        return self

    @staticmethod
    def _extract_offset(string: str) -> Tuple[str, Optional[str]]:
        if 'Z' in string:
            return string.split('Z')[0].strip(), '+0000'
        if '+' in string:
            idx = string.find('+')
        elif string.count('-') in (1, 3):
            idx = string.rfind('-')
        else:
            return string, None
        return string[:idx].strip(), string[idx:].strip()

    @classmethod
    def _extract_date(
            cls, substring: str, string: str) -> Tuple[str, datetime_.datetime]:
        strptime = datetime_.datetime.strptime
        try:
            style, format_ = cls._lastformatstring
            return style, strptime(substring, format_)
        except ValueError:
            for (style, format_) in cls.formatstrings.items():
                for dummy in range(4):
                    try:
                        datetime = strptime(substring, format_)
                        cls._lastformatstring = style, format_
                        return style, datetime
                    except ValueError:
                        format_ = format_[:-3]
            raise ValueError(
                f'The given string `{string}` does not agree '
                f'with any of the supported format styles.')

    @staticmethod
    def _modify_date(date: datetime_.datetime, offset: str, string: str) \
            -> datetime_.datetime:
        try:
            if offset is None:
                return date
            factor = 1 if (offset[0] == '+') else -1
            offset = offset[1:].strip().replace(':', '')
            if len(offset) <= 2:
                minutes = int(offset)*60
            elif len(offset) <= 4:
                minutes = int(offset[:-2])*60 + int(offset[-2:])
            else:
                raise ValueError(
                    'wrong number of offset characters')
            delta = datetime_.timedelta(
                minutes=factor*minutes-hydpy.pub.options.utcoffset)
            return date - delta
        except BaseException:
            raise objecttools.augment_excmessage(
                f'While trying to apply the time zone offset '
                f'defined by string `{string}`')

    @classmethod
    def from_array(cls, array: numpy.ndarray) -> 'Date':
        """Return a |Date| instance based on date information (year,
        month, day, hour, minute, second) stored as the first entries of
        the successive rows of a |numpy.ndarray|.

        >>> from hydpy import Date
        >>> import numpy
        >>> array1d = numpy.array([1992, 10, 8, 15, 15, 42, 999])
        >>> Date.from_array(array1d)
        Date('1992-10-08 15:15:42')

        >>> array3d = numpy.zeros((7, 2, 2))
        >>> array3d[:, 0, 0] = array1d
        >>> Date.from_array(array3d)
        Date('1992-10-08 15:15:42')

        .. note::

           The date defined by the given |numpy.ndarray| cannot
           include any time zone information and corresponds to
           |Options.utcoffset|, which defaults to UTC+01:00.
        """
        intarray = numpy.array(array, dtype=int)
        for dummy in range(1, array.ndim):
            intarray = intarray[:, 0]
        return cls.from_datetime(datetime_.datetime(*intarray[:6]))

    def to_array(self) -> numpy.ndarray:
        """Return a 1-dimensional |numpy| |numpy.ndarray|  with six entries
        defining the actual date (year, month, day, hour, minute, second).

        >>> from hydpy import Date
        >>> Date('1992-10-8 15:15:42').to_array()
        array([ 1992.,    10.,     8.,    15.,    15.,    42.])

        .. note::

           The date defined by the returned |numpy.ndarray| does not
           include any time zone information and corresponds to
           |Options.utcoffset|, which defaults to UTC+01:00.
        """
        return numpy.array([self.year, self.month, self.day, self.hour,
                            self.minute, self.second], dtype=float)

    @classmethod
    def from_cfunits(cls, units: str) -> 'Date':
        """Return a |Date| object representing the reference date of the
        given `units` string agreeing with the NetCDF-CF conventions.

        We took the following example string from the `Time Coordinate`_
        chapter of the NetCDF-CF conventions documentation (modified).  Note
        that method |Date.from_cfunits| ignores the first entry (the unit):

        >>> from hydpy import Date
        >>> Date.from_cfunits('seconds since 1992-10-8 15:15:42 -6:00')
        Date('1992-10-08 22:15:42')
        >>> Date.from_cfunits(' day since 1992-10-8 15:15:00')
        Date('1992-10-08 15:15:00')
        >>> Date.from_cfunits('seconds since 1992-10-8 -6:00')
        Date('1992-10-08 07:00:00')
        >>> Date.from_cfunits('m since 1992-10-8')
        Date('1992-10-08 00:00:00')

        One can also pass the unmodified the example string from
        `Time Coordinate`_, as long as one omits any decimal fractions
        of a second different from zero:

        >>> Date.from_cfunits('seconds since 1992-10-8 15:15:42.')
        Date('1992-10-08 15:15:42')
        >>> Date.from_cfunits('seconds since 1992-10-8 15:15:42.00')
        Date('1992-10-08 15:15:42')
        >>> Date.from_cfunits('seconds since 1992-10-8 15:15:42. -6:00')
        Date('1992-10-08 22:15:42')
        >>> Date.from_cfunits('seconds since 1992-10-8 15:15:42.0 -6:00')
        Date('1992-10-08 22:15:42')
        >>> Date.from_cfunits('seconds since 1992-10-8 15:15:42.005 -6:00')
        Traceback (most recent call last):
        ...
        ValueError: While trying to parse the date of the NetCDF-CF "units" \
string `seconds since 1992-10-8 15:15:42.005 -6:00`, the following error \
occurred: No other decimal fraction of a second than "0" allowed.
        """
        try:
            string = units[units.find('since')+6:]
            idx = string.find('.')
            if idx != -1:
                jdx = None
                for jdx, char in enumerate(string[idx+1:]):
                    if not char.isnumeric():
                        break
                    if char != '0':
                        raise ValueError(
                            'No other decimal fraction of a second '
                            'than "0" allowed.')
                else:
                    if jdx is None:
                        jdx = idx+1
                    else:
                        jdx += 1
                string = f'{string[:idx]}{string[idx+jdx+1:]}'
            return cls.from_string(string)
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to parse the date of the NetCDF-CF "units" '
                f'string `{units}`')

    def to_cfunits(self, unit: str = 'hours', utcoffset: Optional[int] = None) \
            -> str:
        """Return a `units` string agreeing with the NetCDF-CF conventions.

        By default, method |Date.to_cfunits| uses `hours` as the time unit
        and takes the value of |Options.utcoffset| as time zone information:

        >>> from hydpy import Date
        >>> date = Date('1992-10-08 15:15:42')
        >>> date.to_cfunits()
        'hours since 1992-10-08 15:15:42 +01:00'

        You can define arbitrary strings to describe the time unit:

        >>> date.to_cfunits(unit='minutes')
        'minutes since 1992-10-08 15:15:42 +01:00'

        For changing the time zone, pass the corresponding offset in minutes:

        >>> date.to_cfunits(unit='sec', utcoffset=-60)
        'sec since 1992-10-08 13:15:42 -01:00'
        """
        if utcoffset is None:
            utcoffset = hydpy.pub.options.utcoffset
        string = self.to_string('iso2', utcoffset)
        string = ' '.join((string[:-6], string[-6:]))
        return f'{unit} since {string}'

    @property
    def style(self) -> str:
        """Date format style to be applied in printing.

        Initially, |Date.style| corresponds to the format style of the
        string used as the initialisation object of a |Date| object:

        >>> from hydpy import Date
        >>> date = Date('01.11.1997 00:00:00')
        >>> date.style
        'din1'
        >>> date
        Date('01.11.1997 00:00:00')

        However, you are allowed to change it:

        >>> date.style = 'iso1'
        >>> date
        Date('1997-11-01T00:00:00')

        The default style is `iso2`:

        >>> from datetime import datetime
        >>> date = Date(datetime(2000, 1, 1))
        >>> date
        Date('2000-01-01 00:00:00')
        >>> date.style
        'iso2'

        To try to set a non-existing style results in the following
        error message:

        >>> date.style = 'iso'
        Traceback (most recent call last):
        ...
        AttributeError: Date format style `iso` is not available.
        """
        return vars(self).get('style', 'iso2')

    @style.setter
    def style(self, style: str) -> None:
        if style in self.formatstrings:
            vars(self)['style'] = style
        else:
            vars(self).pop('style', None)
            raise AttributeError(
                f'Date format style `{style}` is not available.')

    def _set_thing(self, thing: str, value: int) -> None:
        """Convenience method for `year.fset`, `month.fset`..."""
        try:
            kwargs = {}
            for unit in ('year', 'month', 'day', 'hour', 'minute', 'second'):
                kwargs[unit] = getattr(self, unit)
            kwargs[thing] = int(value)
            self.datetime = datetime_.datetime(**kwargs)
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to change the {thing} '
                f'of the current Date object')

    @property
    def second(self) -> int:
        """The actual second.

        >>> from hydpy import Date
        >>> date = Date('2000-01-01 00:00:00')
        >>> date.second
        0
        >>> date.second = 30
        >>> date.second
        30
        """
        return self.datetime.second

    @second.setter
    def second(self, second: int) -> None:
        self._set_thing('second', second)

    @property
    def minute(self) -> int:
        """The actual minute.

        >>> from hydpy import Date
        >>> date = Date('2000-01-01 00:00:00')
        >>> date.minute
        0
        >>> date.minute = 30
        >>> date.minute
        30
        """
        return self.datetime.minute

    @minute.setter
    def minute(self, minute: int) -> None:
        self._set_thing('minute', minute)

    @property
    def hour(self) -> int:
        """The actual hour.

        >>> from hydpy import Date
        >>> date = Date('2000-01-01 00:00:00')
        >>> date.hour
        0
        >>> date.hour = 12
        >>> date.hour
        12
        """
        return self.datetime.hour

    @hour.setter
    def hour(self, hour: int) -> None:
        self._set_thing('hour', hour)

    @property
    def day(self) -> int:
        """The actual day.

        >>> from hydpy import Date
        >>> date = Date('2000-01-01 00:00:00')
        >>> date.day
        1
        >>> date.day = 15
        >>> date.day
        15
        """
        return self.datetime.day

    @day.setter
    def day(self, day: int) -> None:
        self._set_thing('day', day)

    @property
    def month(self) -> int:
        """The actual month.

        >>> from hydpy import Date
        >>> date = Date('2000-01-01 00:00:00')
        >>> date.month
        1
        >>> date.month = 7
        >>> date.month
        7
        """
        return self.datetime.month

    @month.setter
    def month(self, month: int) -> None:
        self._set_thing('month', month)

    @property
    def year(self) -> int:
        """The actual year.

        >>> from hydpy import Date
        >>> date = Date('2000-01-01 00:00:00')
        >>> date.year
        2000
        >>> date.year = 1   # smallest possible value
        >>> date.year
        1
        >>> date.year = 9999   # highest possible value
        >>> date.year
        9999
        """
        return self.datetime.year

    @year.setter
    def year(self, year: int) -> None:
        self._set_thing('year', year)

    @property
    def refmonth(self) -> int:
        """The first month of the hydrological year.

        The default value is 11 (November which is the German reference month):

        >>> from hydpy import Date
        >>> date1 = Date('2000-01-01')
        >>> date1.refmonth
        11

        Setting it, for example, to 10 (October is another typical reference
        month in different countries) affects all |Date| instances, no
        matter if already existing of if created afterwards:

        >>> date2 = Date('2010-01-01')
        >>> date1.refmonth = 10
        >>> date1.refmonth
        10
        >>> date2.refmonth
        10
        >>> Date('2010-01-01').refmonth
        10

        Alternatively, you can pass an appropriate string (the first
        three characters count):

        >>> date1.refmonth = 'January'
        >>> date1.refmonth
        1
        >>> date1.refmonth = 'feb'
        >>> date1.refmonth
        2

        Wrong arguments result in the following error messages:

        >>> date1.refmonth = 0
        Traceback (most recent call last):
        ...
        ValueError: The reference month must be a value between one \
(January) and twelve (December) but `0` is given

        >>> date1.refmonth = 'wrong'
        Traceback (most recent call last):
        ...
        ValueError: The given argument `wrong` cannot be interpreted as a month.

        >>> date1.refmonth = 11
        """
        return type(self)._firstmonth_wateryear

    @refmonth.setter
    def refmonth(self, value: Union[int, str]):
        try:
            refmonth = int(value)
        except ValueError:
            string = str(value)[:3].lower()
            try:
                months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                          'jul', 'aug', 'sew', 'oct', 'nov', 'dec']
                refmonth = months.index(string) + 1
            except ValueError:
                raise ValueError(
                    f'The given argument `{value}` cannot be '
                    f'interpreted as a month.')
        if not 0 < refmonth < 13:
            raise ValueError(
                f'The reference month must be a value between one '
                f'(January) and twelve (December) but `{value}` is given')
        type(self)._firstmonth_wateryear = refmonth

    @property
    def wateryear(self) -> int:
        """The actual hydrological year according to the selected
        reference month.

        Property |Date.refmonth| defaults to November:

        >>> october = Date('1996.10.01')
        >>> november = Date('1996.11.01')
        >>> october.wateryear
        1996
        >>> november.wateryear
        1997

        Note that changing |Date.refmonth| affects all |Date| objects:

        >>> october.refmonth = 10
        >>> october.wateryear
        1997
        >>> november.wateryear
        1997
        >>> october.refmonth = 'November'
        >>> october.wateryear
        1996
        >>> november.wateryear
        1997
        """
        if self.month < self._firstmonth_wateryear:
            return self.year
        return self.year + 1

    @property
    def dayofyear(self) -> int:
        """The day of the year as an integer value.

        >>> from hydpy import Date
        >>> Date('2003-03-01').dayofyear
        60
        >>> Date('2004-03-01').dayofyear
        61
        """
        return self.datetime.timetuple().tm_yday

    @property
    def leapyear(self) -> bool:
        """Return whether the actual date falls in a leap year or not.

        >>> from hydpy import Date
        >>> Date('2003-03-01').leapyear
        False
        >>> Date('2004-03-01').leapyear
        True
        >>> Date('2000-03-01').leapyear
        True
        >>> Date('2100-03-01').leapyear
        False
        """
        year = self.year
        return (((year % 4) == 0) and
                (((year % 100) != 0) or ((year % 400) == 0)))

    def __add__(self, other: PeriodConstrArg) -> 'Date':
        new = self.from_datetime(self.datetime + Period(other).timedelta)
        new.style = self.style
        return new

    def __iadd__(self, other: PeriodConstrArg) -> 'Date':
        self.datetime += Period(other).timedelta
        return self

    @overload
    def __sub__(self, other: DateConstrArg) -> 'Period':
        """Determine the period between two dates."""

    @overload
    def __sub__(self, other: PeriodConstrArg) -> 'Date':
        """Subtract a period from the actual date."""

    def __sub__(self, other):
        if isinstance(other, (Date, datetime_.datetime, str)):
            try:
                return Period(self.datetime-type(self)(other).datetime)
            except BaseException:
                pass
        if isinstance(other, (Period, datetime_.timedelta, str)):
            try:
                new = self.from_datetime(self.datetime-Period(other).timedelta)
                new.style = self.style
                return new
            except BaseException:
                pass
        raise TypeError(
            f'Object `{other}` of type `{objecttools.classname(other)}` '
            f'cannot be substracted from a `Date` instance.')

    def __rsub__(self, other: DateConstrArg) -> 'Period':
        return Period(type(self)(other).datetime-self.datetime)

    def __isub__(self, other: PeriodConstrArg) -> 'Date':
        self.datetime -= Period(other).timedelta
        return self

    def __lt__(self, other: DateConstrArg) -> bool:
        return self.datetime < type(self)(other).datetime

    def __le__(self, other: DateConstrArg) -> bool:
        return self.datetime <= type(self)(other).datetime

    def __eq__(self, other: Any) -> bool:
        try:
            return self.datetime == type(self)(other).datetime
        except BaseException:
            return False

    def __ne__(self, other: Any) -> bool:
        try:
            return self.datetime != type(self)(other).datetime
        except BaseException:
            return True

    def __gt__(self, other: DateConstrArg) -> bool:
        return self.datetime > type(self)(other).datetime

    def __ge__(self, other: DateConstrArg) -> bool:
        return self.datetime >= type(self)(other).datetime

    def __deepcopy__(self, dict_) -> 'Date':
        new = type(self).from_date(self)
        new.datetime = copy.deepcopy(self.datetime)
        return new

    def to_string(self, style: Optional[str] = None,
                  utcoffset: Optional[int] = None) -> str:
        """Return a |str| object, representing the actual date following
        the given style and the eventually given UTC offset (in minutes).

        Without any input arguments, the actual |Date.style| is used
        to return a date string in your local time zone:

        >>> from hydpy import Date
        >>> date = Date('01.11.1997 00:00:00')
        >>> date.to_string()
        '01.11.1997 00:00:00'

        Passing a style string affects the returned |str| object, but
        not the |Date.style| property:

        >>> date.style
        'din1'
        >>> date.to_string(style='iso2')
        '1997-11-01 00:00:00'
        >>> date.style
        'din1'

        When passing the `utcoffset` in minutes, method |Date.to_string|
        appends the offset string:

        >>> date.to_string(style='iso2', utcoffset=60)
        '1997-11-01 00:00:00+01:00'

        If the given offset does not correspond to your local offset
        defined by |Options.utcoffset| (which defaults to UTC+01:00),
        the date string is adapted:

        >>> date.to_string(style='iso1', utcoffset=0)
        '1997-10-31T23:00:00+00:00'
        """
        if style is None:
            style = self.style
        if utcoffset is None:
            string = ''
            date = self.datetime
        else:
            sign = '+' if utcoffset >= 0 else '-'
            hours = abs(utcoffset // 60)
            minutes = abs(utcoffset % 60)
            string = f'{sign}{hours:02d}:{minutes:02d}'
            offset = utcoffset-hydpy.pub.options.utcoffset
            date = self.datetime + datetime_.timedelta(minutes=offset)
        return date.strftime(self.formatstrings[style]) + string

    def to_repr(self, style: Optional[str] = None,
                utcoffset: Optional[int] = None) -> str:
        """Similar as method |Date.to_string|, but returns a proper
        string representation instead.

        See method |Date.to_string| for explanations on the following
        examples:

        >>> from hydpy import Date
        >>> date = Date('01.11.1997 00:00:00')
        >>> date.to_repr()
        "Date('01.11.1997 00:00:00')"
        >>> date.to_repr('iso1', utcoffset=0)
        "Date('1997-10-31T23:00:00+00:00')"
        """
        return f"Date('{self.to_string(style, utcoffset)}')"

    def __str__(self) -> str:
        return self.to_string(self.style)

    def __repr__(self) -> str:
        return self.to_repr()


class Period:
    """Handles a single period.

    We built the class |Period| on top of the Python module |datetime|.
    In essence, it wraps |datetime.timedelta| objects and specialises
    this general class on the needs of *HydPy* users.

    Be aware of the different minimum time resolution of module |datetime|
    (microseconds) and module |timetools| (seconds).

    You can initialise |Period| directly via |datetime.timedelta|
    objects (see the documentation on method |Period.from_timedelta|
    for more information):

    >>> from hydpy import Period
    >>> from datetime import timedelta
    >>> Period(timedelta(1))
    Period('1d')

    Alternatively, one can initialise from |str| objects.  These must
    consist of some characters defining an integer value followed
    by a single character defining the unit (see the documentation on
    method |Period.from_timedelta| for more information):

    >>> Period('30s')
    Period('30s')

    In case you need an "empty" period object, pass nothing or |None|:

    >>> Period()
    Period()
    >>> Period(None)
    Period()

    All other types result in the following error:

    >>> Period(1)
    Traceback (most recent call last):
    ...
    TypeError: While trying to initialise a `Period` object based \
argument `1`, the following error occurred: The supplied argument \
must be either an instance of `Period`, `datetime.timedelta`, \
or `str`, but the given type is `int`.

    Class |Period| supports some mathematical operations.  Depending
    on the operation, the second operand can be either a number or
    an object interpretable as a date or period.

    First, one can add two |Period| objects or add a |Period| object
    to an object representing a date:

    >>> period = Period('1m')
    >>> period + '2m'
    Period('3m')
    >>> '30s' + period
    Period('90s')
    >>> period += '4m'
    >>> period
    Period('5m')
    >>> '2000-01-01' + period
    Date('2000-01-01 00:05:00')
    >>> period + 'wrong'
    Traceback (most recent call last):
    ...
    TypeError: Object `wrong` of type `str` cannot be added \
to a `Period` instance.

    Subtraction works much alike addition:

    >>> period = Period('4d')
    >>> period - '1d'
    Period('3d')
    >>> '1d' - period
    Period('-3d')
    >>> period -= '2d'
    >>> period
    Period('2d')
    >>> '2000-01-10' - period
    Date('2000-01-08 00:00:00')
    >>> 'wrong' - period
    Traceback (most recent call last):
    ...
    TypeError: A `Period` instance cannot be subtracted \
from object `wrong` of type `str`.

    Use multiplication with a number to change the length of a |Period| object:

    >>> period * 2.0
    Period('4d')
    >>> 0.5 * period
    Period('1d')
    >>> period *= 1.5
    >>> period
    Period('3d')

    Division is possible in combination numbers and objects interpretable
    as periods:


    >>> period / 3.0
    Period('1d')
    >>> period / '36h'
    2.0
    >>> '6d' / period
    2.0
    >>> period /= 1.5
    >>> period
    Period('2d')

    Floor division and calculation of the remainder are also supported:

    >>> period // '20h'
    2
    >>> period % '20h'
    Period('8h')
    >>> '3d' // period
    1
    >>> timedelta(3) % period
    Period('1d')

    You can change the sign in the following manners:

    >>> period = -period
    >>> period
    Period('-2d')
    >>> +period
    Period('-2d')
    >>> abs(period)
    Period('2d')

    The comparison operators work as expected:

    >>> p1, p3 = Period('1d'), Period('3d')
    >>> p1 < '2d', p1 < '1d', '2d' < p1
    (True, False, False)
    >>> p1 <= p3, p1 <= '1d', '2d' <= p1
    (True, True, False)
    >>> p1 == p3, p1 == '1d', '2d' == p1, p1 == '2000-01-01'
    (False, True, False, False)
    >>> p1 != p3, p1 != '1d', '2d' != p1, p1 != '2000-01-01'
    (True, False, True, True)
    >>> p1 >= p3, p1 >= '1d', '2d' >= p1
    (False, True, True)
    >>> p1 > p3, p1 > '1d', '2d' > p1
    (False, False, True)
    """

    def __new__(cls, period: Optional[PeriodConstrArg] = None) -> 'Period':
        try:
            if isinstance(period, Period):
                return cls.from_period(period)
            if isinstance(period, datetime_.timedelta):
                return cls.from_timedelta(period)
            if isinstance(period, str):
                return cls.from_string(period)
            if period is None:
                return super().__new__(cls)
            raise TypeError(
                f'The supplied argument must be either an instance of '
                f'`Period`, `datetime.timedelta`, or `str`, but the '
                f'given type is `{objecttools.classname(period)}`.')
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to initialise a `Period` '
                f'object based argument `{period}`')

    @classmethod
    def from_period(cls, period: 'Period') -> 'Period':
        """Create a new |Period| object based on another |Period| object and
        return it.

        >>> from hydpy import Period
        >>> p1 = Period('1d')
        >>> p2 = Period.from_period(p1)
        >>> p2
        Period('1d')
        >>> p1 *= 2
        >>> p1
        Period('2d')
        >>> p2
        Period('1d')
        """
        self = super().__new__(cls)
        vars(self)['timedelta'] = vars(period).get('timedelta')
        return self

    @classmethod
    def from_timedelta(cls, period: datetime_.timedelta) -> 'Period':
        """Create a new |Period| object based on a |datetime.timedelta|
        object and return it.

        |datetime.timedelta| objects defining days or seconds are allowed,
        but |datetime.timedelta| objects defining microseconds are not:

        >>> from hydpy import Period
        >>> from datetime import timedelta
        >>> Period.from_timedelta(timedelta(1, 0))
        Period('1d')
        >>> Period.from_timedelta(timedelta(0, 1))
        Period('1s')
        >>> Period.from_timedelta(timedelta(0, 0, 1))
        Traceback (most recent call last):
        ...
        ValueError: For `Period` instances, microseconds must be zero.  \
However, for the given `timedelta` object it is `1` instead.
        """
        # pylint: disable=protected-access
        # due to pylint issue https://github.com/PyCQA/pylint/issues/1159
        self = super().__new__(cls)
        vars(self)['timedelta'] = self._check_timedelta(period)
        return self

    @staticmethod
    def _check_timedelta(period: datetime_.timedelta) \
            -> datetime_.timedelta:
        if period.microseconds:
            raise ValueError(
                f'For `Period` instances, microseconds must be zero.  '
                f'However, for the given `timedelta` object it is '
                f'`{period.microseconds}` instead.')
        return period

    @classmethod
    def from_string(cls, period: str) -> 'Period':
        """Create a new |Period| object based on a |str| object and return it.

        The string must consist of a leading integer number followed by
        one of the lower chase characters `s` (seconds), `m` (minutes),
        `h` (hours), and `d` (days):

        >>> from hydpy import Period
        >>> Period.from_string('30s')
        Period('30s')
        >>> Period.from_string('5m')
        Period('5m')
        >>> Period.from_string('6h')
        Period('6h')
        >>> Period.from_string('1d')
        Period('1d')

        Ill-defined strings result in the following errors:

        >>> Period.from_string('oned')
        Traceback (most recent call last):
        ...
        ValueError: All characters of the given period string, \
except the last one which represents the unit, need to define \
an integer number.  Instead, these characters are `one`.

        >>> Period.from_string('1.5d')
        Traceback (most recent call last):
        ...
        ValueError: All characters of the given period string, \
except the last one which represents the unit, need to define \
an integer number.  Instead, these characters are `1.5`.

        >>> Period.from_string('1D')
        Traceback (most recent call last):
        ...
        ValueError: The last character of the given period string needs \
to be either `d` (days), `h` (hours), `m` (minutes),  or `s` (seconds).  \
Instead, the last character is `D`.
        """
        self = super().__new__(cls)
        vars(self)['timedelta'] = cls._get_timedelta_from_string(period)
        return self

    @staticmethod
    def _get_timedelta_from_string(period):
        try:
            number = float(period[:-1])
            if number != int(number):
                raise ValueError
        except ValueError:
            raise ValueError(
                f'All characters of the given period string, '
                f'except the last one which represents the unit, '
                f'need to define an integer number.  Instead, '
                f'these characters are `{period[:-1]}`.')
        unit = period[-1]
        if unit == 'd':
            return datetime_.timedelta(number, 0)
        if unit == 'h':
            return datetime_.timedelta(0, number * 3600)
        if unit == 'm':
            return datetime_.timedelta(0, number * 60)
        if unit == 's':
            return datetime_.timedelta(0, number)
        raise ValueError(
            f'The last character of the given period string needs to '
            f'be either `d` (days), `h` (hours), `m` (minutes),  or `s` '
            f'(seconds).  Instead, the last character is `{unit}`.')

    @classmethod
    def from_seconds(cls, seconds: int) -> 'Period':
        """Create a new |Period| object based on the given integer number
        of seconds and return it.

        >>> from hydpy import Period
        >>> Period.from_seconds(120)
        Period('2m')
        """
        return cls.from_timedelta(datetime_.timedelta(0, int(seconds)))

    @classmethod
    def from_cfunits(cls, units):
        """Create a |Period| object representing the time unit of the
        given `units` string agreeing with the NetCDF-CF conventions
        and return it.

        We took the following example string from the `Time Coordinate`_
        chapter of the NetCDF-CF conventions.  Note that the character
        of the first entry (the actual time unit) is of relevance:

        >>> from hydpy import Period
        >>> Period.from_cfunits('seconds since 1992-10-8 15:15:42.5 -6:00')
        Period('1s')
        >>> Period.from_cfunits(' day since 1992-10-8 15:15:00')
        Period('1d')
        >>> Period.from_cfunits('m since 1992-10-8')
        Period('1m')
        """
        return cls.from_string(f'1{units.strip()[0]}')

    @property
    def timedelta(self) -> datetime_.timedelta:
        """The handled |datetime.timedelta| object.

        You are allowed to change and delete the handled |datetime.timedelta|
        object:

        >>> from hydpy import Period
        >>> period = Period('1d')
        >>> period.timedelta.days
        1
        >>> del period.timedelta
        >>> period.timedelta
        Traceback (most recent call last):
        ...
        AttributeError: The Period object does not handle a timedelta \
object at the moment.
        >>> from datetime import timedelta
        >>> period.timedelta = timedelta(1)
        >>> hasattr(period, 'timedelta')
        True

        Property |Period.timedelta| supports the automatic conversion
        of given |Period| and |str| objects:

        >>> period.timedelta = Period('2d')
        >>> period.timedelta.days
        2

        >>> period.timedelta = '1h'
        >>> period.timedelta.seconds
        3600

        >>> period.timedelta = 1
        Traceback (most recent call last):
        ...
        TypeError: The supplied argument must be either an instance of \
`Period´, `datetime.timedelta` or `str`.  The given arguments type is `int`.
        """
        timedelta = vars(self).get('timedelta')
        if timedelta is None:
            raise AttributeError(
                'The Period object does not handle a '
                'timedelta object at the moment.')
        return timedelta

    @timedelta.setter
    def timedelta(self, period: Optional[PeriodConstrArg]) -> None:
        if isinstance(period, Period):
            vars(self)['timedelta'] = vars(period).get('timedelta')
        elif isinstance(period, datetime_.timedelta):
            vars(self)['timedelta'] = self._check_timedelta(period)
        elif isinstance(period, str):
            vars(self)['timedelta'] = self._get_timedelta_from_string(period)
        else:
            raise TypeError(
                f'The supplied argument must be either an instance '
                f'of `Period´, `datetime.timedelta` or `str`.  The given '
                f'arguments type is `{objecttools.classname(period)}`.')

    @timedelta.deleter
    def timedelta(self) -> None:
        vars(self)['timedelta'] = None

    @property
    def unit(self) -> str:
        """The (most suitable) unit for the current period.

        |Period.unit| always returns the unit leading to the smallest
        integer value:

        >>> from hydpy import Period
        >>> period = Period('1d')
        >>> period.unit
        'd'
        >>> period /= 2
        >>> period.unit
        'h'
        >>> Period('120s').unit
        'm'
        >>> Period('90s').unit
        's'
        """
        if not self.days % 1:
            return 'd'
        if not self.hours % 1:
            return 'h'
        if not self.minutes % 1:
            return 'm'
        return 's'

    @property
    def seconds(self) -> float:
        """Period length in seconds.

        >>> from hydpy import Period
        >>> Period('2d').seconds
        172800.0
        """
        return self.timedelta.total_seconds()

    @property
    def minutes(self) -> float:
        """Period length in minutes.

        >>> from hydpy import Period
        >>> Period('2d').minutes
        2880.0
        """
        return self.timedelta.total_seconds() / 60

    @property
    def hours(self) -> float:
        """Period length in hours.

        >>> from hydpy import Period
        >>> Period('2d').hours
        48.0
        """
        return self.timedelta.total_seconds() / 3600

    @property
    def days(self) -> float:
        """Period length in days.

        >>> from hydpy import Period
        >>> Period('2d').days
        2.0
        """
        return self.timedelta.total_seconds() / 86400

    def __bool__(self) -> bool:
        return bool(getattr(self, 'timedelta', None))

    @overload
    def __add__(self, other: DateConstrArg) -> 'Period':
        """Add the |Period| object to a |Date| object."""

    @overload
    def __add__(self, other: PeriodConstrArg) -> 'Date':
        """Add the |Period| object to another |Period| object."""

    def __add__(self, other):
        if isinstance(other, (Date, datetime_.datetime, str)):
            try:
                other = Date(other)
                new = Date(other.datetime + self.timedelta)
                new.style = other.style
                return new
            except BaseException:
                pass
        if isinstance(other, (Period, datetime_.timedelta, str)):
            try:
                return Period.from_timedelta(
                    self.timedelta + Period(other).timedelta)
            except BaseException:
                pass
        raise TypeError(
            f'Object `{other}` of type `{objecttools.classname(other)}` '
            f'cannot be added to a `Period` instance.')

    @overload
    def __radd__(self, other: DateConstrArg) -> 'Period':
        """Add the |Period| object to a |Date| object."""

    @overload
    def __radd__(self, other: PeriodConstrArg) -> 'Date':
        """Add the |Period| object to another |Period| object."""

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other: PeriodConstrArg) -> 'Period':
        self.timedelta += Period(other).timedelta
        return self

    def __sub__(self, other: PeriodConstrArg) -> 'Period':
        return Period.from_timedelta(self.timedelta - Period(other).timedelta)

    @overload
    def __rsub__(self, other: DateConstrArg) -> 'Period':
        """Subtract the |Period| object from a |Date| object."""

    @overload
    def __rsub__(self, other: PeriodConstrArg) -> 'Date':
        """Subtract the |Period| object from another |Period| object."""

    def __rsub__(self, other):
        if isinstance(other, (Date, datetime_.datetime, str)):
            try:
                other = Date(other)
                new = Date(other.datetime - self.timedelta)
                new.style = other.style
                return new
            except BaseException:
                pass
        if isinstance(other, (Period, datetime_.timedelta, str)):
            try:
                return Period.from_timedelta(
                    Period(other).timedelta - self.timedelta)
            except BaseException:
                pass
        raise TypeError(
            f'A `Period` instance cannot be subtracted from object '
            f'`{other}` of type `{objecttools.classname(other)}`.')

    def __isub__(self, other: PeriodConstrArg) -> 'Period':
        self.timedelta -= Period(other).timedelta
        return self

    def __mul__(self, other: float) -> 'Period':
        return Period.from_timedelta(self.timedelta * other)

    def __rmul__(self, other: float) -> 'Period':
        return self.__mul__(other)

    def __imul__(self, other: float) -> 'Period':
        self.timedelta *= other
        return self

    @overload
    def __truediv__(self, other: PeriodConstrArg) -> float:
        """Divide the |Period| object through another |Period| object."""

    @overload
    def __truediv__(self, other: float) -> 'Period':
        """Divide the |Period| object through a number object."""

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return Period.from_timedelta(self.timedelta / other)
        return self.seconds / Period(other).seconds

    def __rtruediv__(self, other: PeriodConstrArg) -> float:
        return Period(other).seconds / self.seconds

    def __itruediv__(self, other: float) -> 'Period':
        self.timedelta /= other
        return self

    def __floordiv__(self, other: PeriodConstrArg) -> int:
        return self.timedelta // Period(other).timedelta

    def __rfloordiv__(self, other: PeriodConstrArg) -> int:
        return Period(other).timedelta // self.timedelta

    def __mod__(self, other: PeriodConstrArg) -> 'Period':
        return Period.from_timedelta(self.timedelta % Period(other).timedelta)

    def __rmod__(self, other: PeriodConstrArg) -> 'Period':
        return Period.from_timedelta(Period(other).timedelta % self.timedelta)

    def __pos__(self) -> 'Period':
        return Period.from_timedelta(self.timedelta)

    def __neg__(self) -> 'Period':
        return Period.from_timedelta(-self.timedelta)

    def __abs__(self):
        return Period.from_timedelta(abs(self.timedelta))

    def __lt__(self, other: PeriodConstrArg) -> bool:
        return self.timedelta < Period(other).timedelta

    def __le__(self, other: PeriodConstrArg) -> bool:
        return self.timedelta <= Period(other).timedelta

    def __eq__(self, other: Any) -> bool:
        try:
            return self.timedelta == Period(other).timedelta
        except BaseException:
            return False

    def __ne__(self, other: Any) -> bool:
        try:
            return self.timedelta != Period(other).timedelta
        except BaseException:
            return True

    def __gt__(self, other: PeriodConstrArg) -> bool:
        return self.timedelta > Period(other).timedelta

    def __ge__(self, other: PeriodConstrArg) -> bool:
        return self.timedelta >= Period(other).timedelta

    def __str__(self) -> str:
        if self.unit == 'd':
            return f'{self.days:.0f}d'
        if self.unit == 'h':
            return f'{self.hours:.0f}h'
        if self.unit == 'm':
            return f'{self.minutes:.0f}m'
        return f'{self.seconds:.0f}s'

    def __repr__(self) -> str:
        if self:
            return f"Period('{str(self)}')"
        return 'Period()'


class Timegrid:
    """Defines an arbitrary number of equidistant dates via the first date,
    the last date, and the step size between subsequent dates.

    In hydrological modelling, input (and output) data are usually only
    available with a certain resolution, which also determines the possible
    resolution of the actual simulation.  Class |Timegrid| reflects this
    situation by representing equidistant dates.

    To initialise a |Timegrid|, pass its first date, last date and stepsize
    as |str| objects, |Date| and |Period| objects, or |datetime.datetime|
    and |datetime.timedelta| objects (combinations are allowed):

    >>> from hydpy import Date, Period, Timegrid
    >>> timegrid = Timegrid('2000-01-01', '2001-01-01', '1d')
    >>> timegrid
    Timegrid('2000-01-01 00:00:00',
             '2001-01-01 00:00:00',
             '1d')
    >>> timegrid == Timegrid(
    ...     Date('2000-01-01'), Date('2001-01-01'), Period('1d'))
    True
    >>> from datetime import datetime, timedelta
    >>> timegrid == Timegrid(
    ...     datetime(2000, 1, 1), datetime(2001, 1, 1), timedelta(1))
    True

    Passing unsupported argument types results in errors like the following:

    >>> Timegrid('2000-01-01', '2001-01-01', 1)
    Traceback (most recent call last):
    ...
    TypeError: While trying to prepare a Trimegrid object based on the \
arguments `2000-01-01`, `2001-01-01`, and `1`, the following error occurred: \
While trying to initialise a `Period` object based argument `1`, the \
following error occurred: The supplied argument must be either an instance \
of `Period`, `datetime.timedelta`, or `str`, but the given type is `int`.

    You can query indices and the corresponding dates via indexing:

    >>> timegrid[0]
    Date('2000-01-01 00:00:00')
    >>> timegrid[5]
    Date('2000-01-06 00:00:00')
    >>> timegrid[Date('2000-01-01')]
    0
    >>> timegrid['2000-01-06']
    5

    Indexing beyond the ranges of the actual period is allowed:

    >>> timegrid[-365]
    Date('1999-01-01 00:00:00')
    >>> timegrid['2002-01-01']
    731

    However, dates not precisely matching the defined grid result in
    the following error:

    >>> timegrid['2001-01-01 12:00']
    Traceback (most recent call last):
    ...
    ValueError: The given date `2001-01-01 12:00:00` is not properly \
alligned on the indexed timegrid `Timegrid('2000-01-01 00:00:00', \
'2001-01-01 00:00:00', '1d')`.

    You can determine the length of and iterate over |Timegrid| objects:

    >>> len(timegrid)
    366
    >>> for date in timegrid:
    ...     print(date)   # doctest: +ELLIPSIS
    2000-01-01 00:00:00
    2000-01-02 00:00:00
    ...
    2000-12-30 00:00:00
    2000-12-31 00:00:00

    You can check |Timegrid| instances for equality:

    >>> timegrid == Timegrid('2000-01-01', '2001-01-01', '1d')
    True
    >>> timegrid != Timegrid('2000-01-01', '2001-01-01', '1d')
    False
    >>> timegrid == Timegrid('2000-01-02', '2001-01-01', '1d')
    False
    >>> timegrid == Timegrid('2000-01-01', '2001-01-02', '1d')
    False
    >>> timegrid == Timegrid('2000-01-01', '2001-01-01', '2d')
    False
    >>> timegrid == 1
    False

    Also, you can check if a date or even the whole timegrid lies within a
    span defined by a |Timegrid| instance (note unaligned dates and time
    grids with different step sizes are considered unequal):

    >>> Date('2000-01-01') in timegrid
    True
    >>> '2001-01-01' in timegrid
    True
    >>> '2000-07-01' in timegrid
    True
    >>> '1999-12-31' in timegrid
    False
    >>> '2001-01-02' in timegrid
    False
    >>> '2001-01-02 12:00' in timegrid
    False

    >>> timegrid in Timegrid('2000-01-01', '2001-01-01', '1d')
    True
    >>> timegrid in Timegrid('1999-01-01', '2002-01-01', '1d')
    True
    >>> timegrid in Timegrid('2000-01-02', '2001-01-01', '1d')
    False
    >>> timegrid in Timegrid('2000-01-01', '2000-12-31', '1d')
    False
    >>> timegrid in Timegrid('2000-01-01', '2001-01-01', '2d')
    False
    """

    def __init__(
            self,
            firstdate: DateConstrArg,
            lastdate: DateConstrArg,
            stepsize: PeriodConstrArg):
        try:
            self.firstdate = firstdate
            self.lastdate = lastdate
            self.stepsize = stepsize
            self.verify()
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to prepare a Trimegrid object based on the '
                f'arguments `{firstdate}`, `{lastdate}`, and `{stepsize}`')

    @property
    def firstdate(self) -> Date:
        """The start date of the relevant period.

        You can query and alter the value of property |Timegrid.firstdate|
        (call method |Timegrid.verify| afterwards to make sure the |Timegrid|
        object did not become ill-defined):

        >>> from hydpy import Timegrid
        >>> timegrid = Timegrid('2000-01-01', '2001-01-01', '1d')
        >>> timegrid.firstdate
        Date('2000-01-01 00:00:00')
        >>> timegrid.firstdate += '1d'
        >>> timegrid
        Timegrid('2000-01-02 00:00:00',
                 '2001-01-01 00:00:00',
                 '1d')
        """
        return vars(self)['firstdate']

    @firstdate.setter
    def firstdate(self, firstdate: DateConstrArg):
        vars(self)['firstdate'] = Date(firstdate)

    @property
    def lastdate(self) -> Date:
        """The end date of the relevant period.

        You can query and alter the value of property |Timegrid.lastdate|
        (call method |Timegrid.verify| afterwards to make sure the |Timegrid|
        object did not become ill-defined):

        >>> from hydpy import Timegrid
        >>> timegrid = Timegrid('2000-01-01', '2001-01-01', '1d')
        >>> timegrid.lastdate
        Date('2001-01-01 00:00:00')
        >>> timegrid.lastdate += '1d'
        >>> timegrid
        Timegrid('2000-01-01 00:00:00',
                 '2001-01-02 00:00:00',
                 '1d')
        """
        return vars(self)['lastdate']

    @lastdate.setter
    def lastdate(self, lastdate: DateConstrArg) -> None:
        vars(self)['lastdate'] = Date(lastdate)

    @property
    def stepsize(self) -> Period:
        """The time-series data and simulation step size.

        You can query and alter the value of property |Timegrid.stepsize|
        (call method |Timegrid.verify| afterwards to make sure the |Timegrid|
        object did not become ill-defined):

        >>> from hydpy import Timegrid
        >>> timegrid = Timegrid('2000-01-01', '2001-01-01', '1d')
        >>> timegrid.stepsize
        Period('1d')
        >>> timegrid.stepsize += '1d'
        >>> timegrid
        Timegrid('2000-01-01 00:00:00',
                 '2001-01-01 00:00:00',
                 '2d')
        """
        return vars(self)['stepsize']

    @stepsize.setter
    def stepsize(self, stepsize: PeriodConstrArg) -> None:
        vars(self)['stepsize'] = Period(stepsize)

    @classmethod
    def from_array(cls, array: numpy.ndarray) -> 'Timegrid':
        """Create a |Timegrid| instance based on information stored in
        the first 13 rows of a |numpy.ndarray| object and return it.

        In *HydPy*, external time series files do define the time-related
        reference of their data on their own.  For the |numpy| `npy`
        binary format, we achieve this by reserving the first six
        entries for the first date of the period, the next six entries
        for the last date of the period, and the last entry for the step
        size (in seconds):

        >>> from numpy import array
        >>> array_ = array([2000, 1, 1, 0, 0, 0,    # first date
        ...                 2000, 1, 1, 7, 0, 0,    # second date
        ...                 3600,                   # step size (in seconds)
        ...                 1, 2, 3, 4, 5, 6, 7])   # data

        Use method |Timegrid.from_array| to extract the time information:

        >>> from hydpy import Timegrid
        >>> timegrid = Timegrid.from_array(array_)
        >>> timegrid
        Timegrid('2000-01-01 00:00:00',
                 '2000-01-01 07:00:00',
                 '1h')

        Too little information results in the following error message:

        >>> Timegrid.from_array(array_[:12])
        Traceback (most recent call last):
        ...
        IndexError: To define a Timegrid instance via an array, 13 numbers \
are required, but the given array consist of 12 entries/rows only.

        The inverse method |Timegrid.to_array| creates a new |numpy|
        |numpy.ndarray| based on the current |Timegrid| object:

        >>> from hydpy import round_
        >>> round_(timegrid.to_array())
        2000.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2000.0, 1.0, 1.0, 7.0, 0.0, 0.0, 3600.0
        """
        try:
            return cls(Date.from_array(array[:6]),
                       Date.from_array(array[6:12]),
                       Period.from_seconds(array[12].flat[0]))
        except IndexError:
            raise IndexError(
                f'To define a Timegrid instance via an array, 13 '
                f'numbers are required, but the given array '
                f'consist of {len(array)} entries/rows only.')

    def to_array(self) -> numpy.ndarray:
        """Return a 1-dimensional |numpy| |numpy.ndarray| storing the
        information of the actual |Timegrid| object.

        See the documentation on method |Timegrid.from_array| for more
        information.
        """
        values = numpy.empty(13, dtype=float)
        values[:6] = self.firstdate.to_array()
        values[6:12] = self.lastdate.to_array()
        values[12] = self.stepsize.seconds
        return values

    @classmethod
    def from_timepoints(
            cls,
            timepoints: Sequence,
            refdate: DateConstrArg,
            unit: str = 'hours') -> 'Timegrid':
        """Return a |Timegrid| object representing the given starting
        `timepoints` related to the given `refdate`.

        The following examples are identical with the ones of method
        |Timegrid.to_timepoints| but reversed.

        At least two given time points must be increasing and
        equidistant.  By default, they are assumed to be the hours
        elapsed since the given reference date:

        >>> from hydpy import Timegrid
        >>> Timegrid.from_timepoints(
        ...     [0.0, 6.0, 12.0, 18.0], '01.01.2000')
        Timegrid('01.01.2000 00:00:00',
                 '02.01.2000 00:00:00',
                 '6h')
        >>> Timegrid.from_timepoints(
        ...     [24.0, 30.0, 36.0, 42.0], '1999-12-31')
        Timegrid('2000-01-01 00:00:00',
                 '2000-01-02 00:00:00',
                 '6h')

        You can pass other time units (`days` or `min`) explicitly
        (only the first character counts):

        >>> Timegrid.from_timepoints(
        ...     [0.0, 0.25, 0.5, 0.75], '01.01.2000', unit='d')
        Timegrid('01.01.2000 00:00:00',
                 '02.01.2000 00:00:00',
                 '6h')
        >>> Timegrid.from_timepoints(
        ...     [1.0, 1.25, 1.5, 1.75], '1999-12-31', unit='day')
        Timegrid('2000-01-01 00:00:00',
                 '2000-01-02 00:00:00',
                 '6h')
        """
        refdate = Date(refdate)
        unit = Period.from_cfunits(unit)
        delta = timepoints[1]-timepoints[0]
        firstdate = refdate+timepoints[0]*unit
        lastdate = refdate+(timepoints[-1]+delta)*unit
        stepsize = (lastdate-firstdate)/len(timepoints)
        return cls(firstdate, lastdate, stepsize)

    def to_timepoints(
            self,
            unit: str = 'hours',
            offset: Union[float, PeriodConstrArg] = 0.0) -> numpy.ndarray:
        """Return a |numpy.ndarray| representing the starting time points
        of the |Timegrid| object.

        The following examples are identical with the ones of method
        |Timegrid.from_timepoints| but reversed.

        By default, method |Timegrid.to_timepoints| returns the time
        points in hours:

        >>> from hydpy import Timegrid
        >>> timegrid = Timegrid('2000-01-01', '2000-01-02', '6h')
        >>> timegrid.to_timepoints()
        array([  0.,   6.,  12.,  18.])

        You can define other time units (`days` or `min`) (only the first
        character counts):

        >>> timegrid.to_timepoints(unit='d')
        array([ 0.  ,  0.25,  0.5 ,  0.75])

        Additionally, one can pass an `offset` that must be of type |int|
        or a valid |Period| initialisation argument:

        >>> timegrid.to_timepoints(offset=24)
        array([ 24.,  30.,  36.,  42.])
        >>> timegrid.to_timepoints(offset='1d')
        array([ 24.,  30.,  36.,  42.])
        >>> timegrid.to_timepoints(unit='day', offset='1d')
        array([ 1.  ,  1.25,  1.5 ,  1.75])
        """
        unit = Period.from_cfunits(unit)
        if not isinstance(offset, (float, int)):
            offset = Period(offset)/unit
        step = self.stepsize/unit
        nmb = len(self)
        variable = numpy.linspace(offset, offset+step*(nmb-1), nmb)
        return variable

    def array2series(self, array: numpy.ndarray) -> numpy.ndarray:
        """Prefix the information of the actual |Timegrid| object to the
        given array and return it.

        The |Timegrid| information is available in the first thirteen values
        of the first axis of the returned series (see the documentation
        on the method |Timegrid.from_array|).

        To show how method |Timegrid.array2series| works, we first apply
        it on a simple list containing numbers:

        >>> from hydpy import Timegrid
        >>> timegrid = Timegrid('2000-11-01 00:00', '2000-11-01 04:00', '1h')
        >>> series = timegrid.array2series([1, 2, 3.5, '5.0'])

        The first six entries contain the first date of the timegrid (year,
        month, day, hour, minute, second):

        >>> from hydpy import round_
        >>> round_(series[:6])
        2000.0, 11.0, 1.0, 0.0, 0.0, 0.0

        The six subsequent entries contain the last date:

        >>> round_(series[6:12])
        2000.0, 11.0, 1.0, 4.0, 0.0, 0.0

        The thirteens value is the step size in seconds:

        >>> round_(series[12])
        3600.0

        The last four value are the ones of the given vector:

        >>> round_(series[-4:])
        1.0, 2.0, 3.5, 5.0

        The given array can have an arbitrary number of dimensions:

        >>> import numpy
        >>> array = numpy.eye(4)
        >>> series = timegrid.array2series(array)

        Now the timegrid information is stored in the first column:

        >>> round_(series[:13, 0])
        2000.0, 11.0, 1.0, 0.0, 0.0, 0.0, 2000.0, 11.0, 1.0, 4.0, 0.0, 0.0, \
3600.0

        All other columns of the first thirteen rows contain |numpy.nan|
        values:

        >>> round_(series[12, :])
        3600.0, nan, nan, nan

        The original values are available in the last four rows:

        >>> round_(series[13, :])
        1.0, 0.0, 0.0, 0.0

        Inappropriate array objects result in error messages like
        the following:

        >>> timegrid.array2series([[1, 2], [3]])
        Traceback (most recent call last):
        ...
        ValueError: While trying to prefix timegrid information to the given \
array, the following error occurred: setting an array element with a sequence.

        The following error occurs when the given array does not fit to
        the defined time grid.

        >>> timegrid.array2series([[1, 2], [3, 4]])
        Traceback (most recent call last):
        ...
        ValueError: When converting an array to a sequence, the lengths of \
the timegrid and the given array must be equal, but the length of the \
timegrid object is `4` and the length of the array object is `2`.
        """
        try:
            array = numpy.array(array, dtype=float)
        except BaseException:
            objecttools.augment_excmessage(
                'While trying to prefix timegrid information to the '
                'given array')
        if len(array) != len(self):
            raise ValueError(
                f'When converting an array to a sequence, the lengths of the '
                f'timegrid and the given array must be equal, but the length '
                f'of the timegrid object is `{len(self)}` and the length of '
                f'the array object is `{len(array)}`.')
        shape = list(array.shape)
        shape[0] += 13
        series = numpy.full(shape, numpy.nan)
        slices = [slice(0, 13)]
        subshape = [13]
        for dummy in range(1, series.ndim):
            slices.append(slice(0, 1))
            subshape.append(1)
        series[tuple(slices)] = self.to_array().reshape(subshape)
        series[13:] = array
        return series

    def verify(self) -> None:
        """Raise a |ValueError| if the dates or the step size of the
        |Timegrid| object are currently inconsistent.

        Method |Timegrid.verify| is called at the end of the initialisation
        of a new |Timegrid| object automatically:

        >>> from hydpy import Timegrid
        >>> Timegrid('2001-01-01', '2000-01-01', '1d')
        Traceback (most recent call last):
        ...
        ValueError: While trying to prepare a Trimegrid object based on the \
arguments `2001-01-01`, `2000-01-01`, and `1d`, the following error occurred: \
The temporal sequence of the first date (`2001-01-01 00:00:00`) and the last \
date (`2000-01-01 00:00:00`) is inconsistent.

        However, the same does not hold when changing property
        |Timegrid.firstdate|, |Timegrid.lastdate|, or |Timegrid.stepsize|:

        >>> timegrid = Timegrid('2000-01-01', '2001-01-01', '1d')
        >>> timegrid.stepsize = '4d'

        When in doubt, call method |Timegrid.verify| manually:

        >>> timegrid.verify()
        Traceback (most recent call last):
        ...
        ValueError: The interval between the first date \
(`2000-01-01 00:00:00`) and the last date (`2001-01-01 00:00:00`) is \
`366d`, which is not an integral multiple of the step size `4d`.
        """
        if self.firstdate >= self.lastdate:
            raise ValueError(
                f'The temporal sequence of the first date (`{self.firstdate}`) '
                f'and the last date (`{self.lastdate}`) is inconsistent.')
        if (self.lastdate-self.firstdate) % self.stepsize:
            raise ValueError(
                f'The interval between the first date (`{self.firstdate}`) '
                f'and the last date (`{self.lastdate}`) is '
                f'`{self.lastdate-self.firstdate}`, which is not an '
                f'integral multiple of the step size `{self.stepsize}`.')

    def __len__(self) -> int:
        return abs(int((self.lastdate-self.firstdate) / self.stepsize))

    @overload
    def __getitem__(self, key: int) -> Date:
        """Get the date corresponding to the given index value."""

    @overload
    def __getitem__(self, key: DateConstrArg) -> int:
        """Get the index value corresponding to the given date."""

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            return Date(self.firstdate + key*self.stepsize)
        key = Date(key)
        index = (key-self.firstdate) / self.stepsize
        if index % 1.:
            raise ValueError(
                f'The given date `{key}` is not properly alligned on '
                f'the indexed timegrid `{self}`.')
        return int(index)

    def __iter__(self) -> Iterator[Date]:
        dt = copy.deepcopy(self.firstdate).datetime
        last_dt = self.lastdate.datetime
        td = self.stepsize.timedelta
        from_datetime = Date.from_datetime
        while dt < last_dt:
            yield from_datetime(dt)
            dt = dt + td

    def _containsdate(self, date: Date) -> bool:
        return ((self.firstdate <= date <= self.lastdate) and not
                ((date-self.firstdate) % self.stepsize))

    def _containstimegrid(self, timegrid: 'Timegrid') -> bool:
        return (self._containsdate(timegrid.firstdate) and
                self._containsdate(timegrid.lastdate) and
                (timegrid.stepsize == self.stepsize))

    def __contains__(self, other: Union[DateConstrArg, 'Timegrid']):
        if isinstance(other, Timegrid):
            return self._containstimegrid(other)
        return self._containsdate(Date(other))

    def __eq__(self, other: Any) -> bool:
        try:
            return ((self.firstdate == other.firstdate) and
                    (self.lastdate == other.lastdate) and
                    (self.stepsize == other.stepsize))
        except AttributeError:
            return False

    def __repr__(self) -> str:
        return self.assignrepr('')

    def __str__(self) -> str:
        return objecttools.flatten_repr(self)

    def assignrepr(
            self, prefix: str,
            style: Optional[str] = None,
            utcoffset: Optional[int] = None) -> str:
        """Return a |repr| string with a prefixed assignment.

        >>> from hydpy import Timegrid
        >>> timegrid = Timegrid('1996-11-01 00:00:00',
        ...                     '1997-11-01 00:00:00',
        ...                     '1d')
        >>> print(timegrid.assignrepr(prefix='timegrid = '))
        timegrid = Timegrid('1996-11-01 00:00:00',
                            '1997-11-01 00:00:00',
                            '1d')

        >>> print(timegrid.assignrepr(
        ...     prefix='', style='iso1', utcoffset=120))
        Timegrid('1996-11-01T01:00:00+02:00',
                 '1997-11-01T01:00:00+02:00',
                 '1d')
        """
        skip = len(prefix) + 9
        blanks = ' ' * skip
        return (f"{prefix}Timegrid('"
                f"{self.firstdate.to_string(style, utcoffset)}',\n"
                f"{blanks}'{self.lastdate.to_string(style, utcoffset)}',\n"
                f"{blanks}'{str(self.stepsize)}')")


class Timegrids:
    """Handles both the "initialisation" and the "simulation" |Timegrid|
    object of a *HydPy* project.

    The HydPy framework distinguishes two "time frames", one associated
    with the initialisation period (`init`) and one associated with the
    actual simulation period (`sim`).  These time frames are represented
    by two different |Timegrid| objects, which are both handled by a
    single |Timegrids| object.

    There is usually only one |Timegrids| object required within a
    *HydPy* project available as attribute `timegrids` of module |pub|.
    You have to create such an object at the beginning of your workflow.

    In many cases, one wants to perform simulations covering the whole
    initialisation period.  Then you can pass a single |Timegrid|
    instance to the constructor of class |Timegrids|:

    >>> from hydpy import Timegrid, Timegrids
    >>> timegrids = Timegrids(Timegrid(
    ...     '2000-01-01', '2001-01-01', '1d'))
    >>> print(timegrids)
    Timegrids(Timegrid('2000-01-01 00:00:00', '2001-01-01 00:00:00', '1d'))

    An even shorter approach is to pass the arguments of the
    |Timegrid| constructor directly:

    >>> timegrids == Timegrids('2000-01-01', '2001-01-01', '1d')
    True

    Also, you can pass another |Timegrids| object:

    >>> timegrids == Timegrids(timegrids)
    True

    To define a simulation time grid different from the initialisation
    time grid, pass them as two individual |Timegrid| objects:

    >>> timegrids = Timegrids(
    ...     Timegrid('2000-01-01', '2001-01-01', '1d'),
    ...     Timegrid('2000-01-01', '2000-07-01', '1d'))
    >>> timegrids
    Timegrids(Timegrid('2000-01-01 00:00:00',
                       '2001-01-01 00:00:00',
                       '1d'),
              Timegrid('2000-01-01 00:00:00',
                       '2000-07-01 00:00:00',
                       '1d'))
    >>> timegrids.init
    Timegrid('2000-01-01 00:00:00',
             '2001-01-01 00:00:00',
             '1d')
    >>> timegrids.sim
    Timegrid('2000-01-01 00:00:00',
             '2000-07-01 00:00:00',
             '1d')

    Wrong arguments should result in understandable error messages:

    >>> Timegrids(1, 2, 3, 4)
    Traceback (most recent call last):
    ...
    ValueError: While trying to define a new Timegrids object based on \
arguments `1, 2, 3, and 4`, the following error occurred: Initialising \
`Timegrids` objects requires one, two, or three arguments but `4` are given.

    >>> Timegrids('wrong')
    Traceback (most recent call last):
    ...
    TypeError: While trying to define a new Timegrids object based on \
arguments `wrong`, the following error occurred: When passing a single \
argument to the constructor of class `Timegrids`, the argument must be \
a `Timegrid` or a `Timegrids` object, but a `str` is given.

    >>> Timegrids('very', 'wrong')
    Traceback (most recent call last):
    ...
    TypeError: While trying to define a new Timegrids object based on \
arguments `very and wrong`, the following error occurred: When passing \
two arguments to the constructor of class `Timegrids`, both argument \
must be `Timegrid` object, but the first one is of type `str`.

    Two |Timegrids| objects are equal if both the respective initialisation
    and simulation periods are equal:

    >>> timegrids == Timegrids(
    ...     Timegrid('2000-01-01', '2001-01-01', '1d'),
    ...     Timegrid('2000-01-01', '2000-07-01', '1d'))
    True
    >>> timegrids == Timegrids(
    ...     Timegrid('1999-01-01', '2001-01-01', '1d'),
    ...     Timegrid('2000-01-01', '2000-07-01', '1d'))
    False
    >>> timegrids == Timegrids(
    ...     Timegrid('2000-01-01', '2001-01-01', '1d'),
    ...     Timegrid('2000-01-01', '2001-01-01', '1d'))
    False
    >>> timegrids == Date('2000-01-01')
    False
    """

    def __new__(cls, *args: Union[Timegrid, DateConstrArg, PeriodConstrArg]):
        if (len(args) == 1) and isinstance(args[0], Timegrids):
            return args[0]
        return super().__new__(cls)

    def __init__(self, *args: Any):
        try:
            nmbargs = len(args)
            if not nmbargs or nmbargs > 3:
                raise ValueError(
                    f'Initialising `Timegrids` objects requires one, two, '
                    f'or three arguments but `{nmbargs}` are given.')
            if nmbargs == 1:
                if isinstance(args[0], Timegrids):
                    pass
                elif isinstance(args[0], Timegrid):
                    self.init = args[0]
                    self.sim = copy.deepcopy(args[0])
                else:
                    raise TypeError(
                        f'When passing a single argument to the constructor '
                        f'of class `Timegrids`, the argument must be a '
                        f'`Timegrid` or a `Timegrids` object, but a '
                        f'`{objecttools.classname(args[0])}` is given.')
            elif nmbargs == 2:
                for idx, arg in enumerate(args):
                    if not isinstance(arg, Timegrid):
                        number = 'second' if idx else 'first'
                        raise TypeError(
                            f'When passing two arguments to the constructor '
                            f'of class `Timegrids`, both argument must be '
                            f'`Timegrid` object, but the {number} one is of '
                            f'type `{objecttools.classname(args[0])}`.')
                self.init = args[0]
                self.sim = args[1]
            elif nmbargs == 3:
                self.init = Timegrid(args[0], args[1], args[2])
                self.sim = Timegrid(args[0], args[1], args[2])
            self.verify()
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to define a new Timegrids object based on '
                f'arguments `{objecttools.enumeration(args)}`')

    @property
    def stepsize(self) -> Period:
        """Stepsize of all handled |Timegrid| objects.

        You can change the (the identical) |Timegrid.stepsize| of all
        handled |Timegrid| objects at once:

        >>> from hydpy import Timegrids
        >>> timegrids = Timegrids('2000-01-01', '2001-01-01', '1d')
        >>> timegrids.sim.lastdate = '2000-02-01'
        >>> timegrids
        Timegrids(Timegrid('2000-01-01 00:00:00',
                           '2001-01-01 00:00:00',
                           '1d'),
                  Timegrid('2000-01-01 00:00:00',
                           '2000-02-01 00:00:00',
                           '1d'))

        >>> timegrids.stepsize
        Period('1d')
        >>> timegrids.stepsize = '1h'
        >>> timegrids
        Timegrids(Timegrid('2000-01-01 00:00:00',
                           '2001-01-01 00:00:00',
                           '1h'),
                  Timegrid('2000-01-01 00:00:00',
                           '2000-02-01 00:00:00',
                           '1h'))
        """
        return self.init.stepsize

    @stepsize.setter
    def stepsize(self, stepsize: PeriodConstrArg) -> None:
        self.init.stepsize = Period(stepsize)
        self.sim.stepsize = Period(stepsize)

    def verify(self) -> None:
        """Raise a |ValueError| if the different |Timegrid| objects are
        inconsistent.

        Method |Timegrids.verify| is called at the end of the initialisation
        of a new |Timegrids| object automatically:

        >>> from hydpy import Timegrid, Timegrids
        >>> Timegrids(
        ...     Timegrid('2001-01-01', '2002-01-01', '1d'),
        ...     Timegrid('2000-01-01', '2002-01-01', '1d'))
        Traceback (most recent call last):
        ...
        ValueError: While trying to define a new Timegrids object based \
on arguments `Timegrid('2001-01-01 00:00:00', '2002-01-01 00:00:00', '1d') \
and Timegrid('2000-01-01 00:00:00', '2002-01-01 00:00:00', '1d')`, the \
following error occurred: The first date of the initialisation period \
(`2001-01-01 00:00:00`) must not be later than the first date of the \
simulation period (`2000-01-01 00:00:00`).

        However, the same does not hold when one changes the initialisation
        or the simulation time grid later:

        >>> timegrids = Timegrids(
        ...     Timegrid('2001-01-01', '2002-01-01', '1d'),
        ...     Timegrid('2001-01-01', '2002-01-01', '1d'))
        >>> timegrids.sim.lastdate = '2003-01-01'

        When in doubt, call method |Timegrids.verify| manually:

        >>> timegrids.verify()
        Traceback (most recent call last):
        ...
        ValueError: The last date of the initialisation period \
(`2002-01-01 00:00:00`) must not be earlier than the last date \
of the simulation period (`2003-01-01 00:00:00`).

        Besides both tests explained by the above error messages, method
        |Timegrids.verify| checks for an equal step size of both
        |Timegrid| objects and their proper alignment:

        >>> timegrids.sim.lastdate = '2002-01-01'
        >>> timegrids.sim.stepsize = '5d'
        >>> timegrids.verify()
        Traceback (most recent call last):
        ...
        ValueError: The initialisation stepsize (`1d`) must be identical \
with the simulation stepsize (`5d`).

        >>> timegrids.sim = Timegrid(
        ...     '2001-01-01 12:00', '2001-12-31 12:00', '1d')
        >>> timegrids.verify()
        Traceback (most recent call last):
        ...
        ValueError: The simulation time grid `Timegrid('2001-01-01 12:00:00', \
'2001-12-31 12:00:00', '1d')` is not properly alligned on the initialisation \
time grid `Timegrid('2001-01-01 00:00:00', '2002-01-01 00:00:00', '1d')`.

        Additionally, the method |Timegrids.verify| calls the
        verification methods of both |Timegrid| objects:

        >>> timegrids.sim.stepsize = '3d'
        >>> timegrids.verify()
        Traceback (most recent call last):
        ...
        ValueError: While trying to verify the simulation time grid \
`Timegrid('2001-01-01 00:00:00', '2002-01-01 00:00:00', '1d')`, \
the following error occurred: The interval between the first date \
(`2001-01-01 12:00:00`) and the last date (`2001-12-31 12:00:00`) \
is `364d`, which is not an integral multiple of the step size `3d`.

        >>> timegrids.init.stepsize = '3d'
        >>> timegrids.verify()
        Traceback (most recent call last):
        ...
        ValueError: While trying to verify the initialisation time grid \
`Timegrid('2001-01-01 00:00:00', '2002-01-01 00:00:00', '3d')`, \
the following error occurred: The interval between the first date \
(`2001-01-01 00:00:00`) and the last date (`2002-01-01 00:00:00`) \
is `365d`, which is not an integral multiple of the step size `3d`.
        """
        try:
            self.init.verify()
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to verify the initialisation '
                f'time grid `{self.init}`')
        try:
            self.sim.verify()
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to verify the simulation '
                f'time grid `{self.init}`')
        if self.init.firstdate > self.sim.firstdate:
            raise ValueError(
                f'The first date of the initialisation period '
                f'(`{self.init.firstdate}`) must not be later '
                f'than the first date of the simulation period '
                f'(`{self.sim.firstdate}`).')
        if self.init.lastdate < self.sim.lastdate:
            raise ValueError(
                f'The last date of the initialisation period '
                f'(`{self.init.lastdate}`) must not be earlier '
                f'than the last date of the simulation period '
                f'(`{self.sim.lastdate}`).')
        if self.init.stepsize != self.sim.stepsize:
            raise ValueError(
                f'The initialisation stepsize (`{self.init.stepsize}`) '
                f'must be identical with the simulation stepsize '
                f'(`{self.sim.stepsize}`).')
        try:
            self.init[self.sim.firstdate]
        except ValueError:
            raise ValueError(
                f'The simulation time grid `{self.sim}` is not properly '
                f'alligned on the initialisation time grid `{self.init}`.')

    @property
    def simindices(self) -> Tuple[int, int]:
        """A tuple containing the start and end index of the simulation period
        regarding the initialisation period.

        >>> from hydpy import Timegrids
        >>> timegrids = Timegrids('2000-01-01', '2001-01-01', '1d')
        >>> timegrids.simindices
        (0, 366)
        >>> timegrids.sim.firstdate = '2000-01-11'
        >>> timegrids.sim.lastdate = '2000-02-01'
        >>> timegrids.simindices
        (10, 31)
        """
        return self.init[self.sim.firstdate], self.init[self.sim.lastdate]

    def qfactor(self, area: float) -> float:
        """Return the factor for converting `mm/stepsize` to `m³/s` for
        a reference area, given in `km²`.

        >>> from hydpy import Timegrids, round_
        >>> timegrids = Timegrids('2000-01-01', '2001-01-01', '1s')
        >>> timegrids.qfactor(1.0)
        1000.0
        >>> timegrids.stepsize = '2d'
        >>> round_(timegrids.qfactor(2.0))
        0.011574
        """
        return area * 1000. / self.stepsize.seconds

    def parfactor(self, stepsize: PeriodConstrArg) -> float:
        """Return the factor for adjusting time-dependent parameter values
        to the actual simulation step size (the given `stepsize` must be
        related to the original parameter values).

        >>> from hydpy import Timegrids
        >>> timegrids = Timegrids('2000-01-01', '2001-01-01', '1d')
        >>> timegrids.parfactor('1d')
        1.0
        >>> timegrids.parfactor('1h')
        24.0
        """
        return self.stepsize / Period(stepsize)

    def __eq__(self, other: Any) -> bool:
        try:
            return ((self.init == other.init) and
                    (self.sim == other.sim))
        except AttributeError:
            return False

    def __repr__(self) -> str:
        return self.assignrepr('')

    def assignrepr(self, prefix: str) -> str:
        """Return a |repr| string with a prefixed assignment."""
        caller = 'Timegrids('
        blanks = ' ' * (len(prefix) + len(caller))
        prefix = f'{prefix}{caller}'
        lines = [f'{self.init.assignrepr(prefix)},']
        if self.sim != self.init:
            lines.append(f'{self.sim.assignrepr(blanks)},')
        lines[-1] = lines[-1][:-1] + ')'
        return '\n'.join(lines)

    def __str__(self) -> str:
        return objecttools.flatten_repr(self)


class TOY:
    """Time of year handler.

    |TOY| objects are used to define certain things that are true for a
    specific time point in each year.  The smallest supported time unit is
    seconds.

    For initialisation, one usually passes a string defining the month,
    the day, the hour, the minute and the second in the mentioned order
    and separated by single underscores:

    >>> from hydpy.core.timetools import Date, TOY
    >>> t = TOY('3_13_23_33_43')
    >>> t.month
    3
    >>> t.day
    13
    >>> t.hour
    23
    >>> t.minute
    33
    >>> t.second
    43

    If a lower precision is required, one can shorten the string, which
    implicitly sets the omitted property to the lowest possible value:

    >>> TOY('3_13_23_33')
    TOY('3_13_23_33_0')

    The most extreme example is to pass no string at all:

    >>> TOY()
    TOY('1_1_0_0_0')

    One can prefix some information to the string, which is useful when the
    string is to be used as a valid variable name somewhere else:

    >>> TOY('something_3_13_23_33_2')
    TOY('3_13_23_33_2')

    As one can see, we lose the prefixed information in the printed string
    representation.  Instead, applying "str" returns a string with a
    standard prefix:

    >>> str(TOY('something_3_13_23_33_2'))
    'toy_3_13_23_33_2'

    Alternatively, one can use a |Date| object as an initialisation
    argument, omitting the year:

    >>> TOY(Date('2001.02.03 04:05:06'))
    TOY('2_3_4_5_6')

    Ill-defined constructor arguments result in error messages like
    the following:

    >>> TOY('2_30_4_5_6')
    Traceback (most recent call last):
    ...
    ValueError: While trying to initialise a TOY object based on argument \
`value `2_30_4_5_6` of type `str`, the following error occurred: While \
trying to retrieve the day, the following error occurred: The value of \
property `day` of the actual TOY (time of year) object must lie within \
the range `(1, 29)`, as the month has already been set to `2`, but the \
given value is `30`.

    It is only allowed to modify the mentioned properties, not to define new
    ones:

    >>> t.microsecond = 53
    Traceback (most recent call last):
    ...
    AttributeError: TOY (time of year) objects only allow to set the \
properties month, day, hour, minute, and second, but `microsecond` is given.

    You can pass any objects convertible to integers:

    >>> t.second = '53'
    >>> t.second
    53

    Unconvertible objects cause the following error:

    >>> t.second = 'fiftythree'
    Traceback (most recent call last):
    ...
    ValueError: For TOY (time of year) objects, all properties must be of \
type `int`, but the value `fiftythree` of type `str` given for property \
`second` cannot be converted to `int`.

    Additionally, given values are checked to lie within a suitable range:

    >>> t.second = 60
    Traceback (most recent call last):
    ...
    ValueError: The value of property `second` of TOY (time of year) \
objects must lie within the range `(0, 59)`, but the given value is `60`.

    Note that the allowed values for `month` and `day` depend on each other,
    which is why the order one defines them might be of importance.  So, if
    January is predefined, one can set day to the 31st:

    >>> t.month = 1
    >>> t.day = 31

    Afterwards, one cannot directly change the month to April:

    >>> t.month = 4
    Traceback (most recent call last):
    ...
    ValueError: The value of property `month` of the actual TOY \
(time of year) object must not be the given value `4`, as the day \
has already been set to `31`.

    First set `day` to a smaller value and then change `month`:

    >>> t.day = 30
    >>> t.month = 4

    It is possible to compare two |TOY| instances:

    >>> t1, t2 = TOY('1'), TOY('2')
    >>> t1 < t1, t1 < t2, t2 < t1
    (False, True, False)
    >>> t1 <= t1, t1 <= t2, t2 <= t1
    (True, True, False)
    >>> t1 == t1, t1 == t2, t1 == 1
    (True, False, False)
    >>> t1 != t1, t1 != t2, t1 != 1
    (False, True, True)
    >>> t1 >= t1, t1 >= t2, t2 >= t1
    (True, False, True)
    >>> t1 > t1, t1 > t2, t2 > t1
    (False, False, True)

    Subtracting two |TOY| object gives their time difference in seconds:

    >>> TOY('1_1_0_3_0') - TOY('1_1_0_1_30')
    90

    Subtraction never results in negative values, due to assuming the
    left operand is the posterior (eventually within the subsequent year):

    >>> TOY('1_1_0_1_30') - TOY('12_31_23_58_30')
    180
    """
    _PROPERTIES = collections.OrderedDict((('month', (1, 12)),
                                           ('day', (1, 31)),
                                           ('hour', (0, 23)),
                                           ('minute', (0, 59)),
                                           ('second', (0, 59))))
    _STARTDATE = Date.from_datetime(datetime_.datetime(2000, 1, 1))
    _ENDDATE = Date.from_datetime(datetime_.datetime(2001, 1, 1))

    month: int
    day: int
    hour: int
    minute: int
    second: int

    def __init__(self, value: Union[str, Date] = ''):
        try:
            if isinstance(value, Date):
                datetime = value.datetime
                dict_ = vars(self)
                for name in self._PROPERTIES.keys():
                    dict_[name] = getattr(datetime, name)
            else:
                values = value.split('_')
                if not values[0].isdigit():
                    del values[0]
                for prop in self._PROPERTIES:
                    try:
                        setattr(self, prop, values.pop(0))
                    except IndexError:
                        if prop in ('month', 'day'):
                            setattr(self, prop, 1)
                        else:
                            setattr(self, prop, 0)
                    except ValueError:
                        objecttools.augment_excmessage(
                            f'While trying to retrieve the {prop}')
            vars(self)['seconds_passed'] = None
            vars(self)['seconds_left'] = None
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to initialise a TOY object based on '
                f'argument `{objecttools.value_of_type(value)}')

    def __setattr__(self, name: str, value: int) -> None:
        if name not in self._PROPERTIES:
            raise AttributeError(
                f'TOY (time of year) objects only allow to set the '
                f'properties {objecttools.enumeration(self._PROPERTIES.keys())}'
                f', but `{name}` is given.')
        try:
            value = int(value)
        except ValueError:
            raise ValueError(
                f'For TOY (time of year) objects, all properties must be of '
                f'type `int`, but the {objecttools.value_of_type(value)} '
                f'given for property `{name}` cannot be converted to `int`.')
        if (name == 'day') and hasattr(self, 'month'):
            bounds = (1, calendar.monthrange(2000, self.month)[1])
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(
                    f'The value of property `day` of the actual TOY '
                    f'(time of year) object must lie within the range '
                    f'`{bounds}`, as the month has already been set to '
                    f'`{self.month}`, but the given value is `{value}`.')
        elif (name == 'month') and hasattr(self, 'day'):
            bounds = (1, calendar.monthrange(2000, value)[1])
            if not bounds[0] <= self.day <= bounds[1]:
                raise ValueError(
                    f'The value of property `month` of the actual TOY '
                    f'(time of year) object must not be the given value '
                    f'`{value}`, as the day has already been set to '
                    f'`{self.day}`.')
        else:
            bounds = self._PROPERTIES[name]
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(
                    f'The value of property `{name}` of TOY (time of '
                    f'year) objects must lie within the range `{bounds}`, '
                    f'but the given value is `{value}`.')
        super().__setattr__(name, value)
        vars(self)['seconds_passed'] = None
        vars(self)['seconds_left'] = None

    @property
    def seconds_passed(self) -> int:
        """The amount of time passed in seconds since the beginning of the year.

        In the first example, the year is only one minute and thirty seconds
        old:

        >>> from hydpy.core.timetools import TOY
        >>> toy = TOY('1_1_0_1_30')
        >>> toy.seconds_passed
        90

        Updating the |TOY| object triggers a recalculation of property
        |TOY.seconds_passed|:

        >>> toy.day = 2
        >>> toy.seconds_passed
        86490

        The second example shows the general inclusion of the 29th of February:

        >>> TOY('3').seconds_passed
        5184000
        """
        seconds_passed = vars(self)['seconds_passed']
        if seconds_passed is None:
            seconds_passed = int(
                (self._datetime-self._STARTDATE.datetime).total_seconds())
            vars(self)['seconds_passed'] = seconds_passed
        return seconds_passed

    @property
    def seconds_left(self) -> int:
        """The remaining amount of time part of the year in seconds.

        In the first example, only one minute and thirty seconds of the year
        remain:

        >>> from hydpy.core.timetools import TOY
        >>> toy = TOY('12_31_23_58_30')
        >>> toy.seconds_left
        90

        Updating the |TOY| object triggers a recalculation of property
        |TOY.seconds_passed|:

        >>> toy.day = 30
        >>> toy.seconds_left
        86490

        The second example shows the general inclusion of the 29th of February:

        >>> TOY('2').seconds_left
        28944000
        """
        seconds_left = vars(self)['seconds_left']
        if seconds_left is None:
            seconds_left = int(
                (self._ENDDATE.datetime-self._datetime).total_seconds())
            vars(self)['seconds_left'] = seconds_left
        return seconds_left

    @property
    def _datetime(self):
        return datetime_.datetime(
            2000, self.month, self.day, self.hour, self.minute, self.second)

    @classmethod
    def centred_timegrid(cls) -> Tuple[Timegrid, numpy.ndarray]:
        """Return a |Timegrid| object defining the central time points
        of the year 2000 and a boolean array describing its intersection
        with the current initialisation period not taking the year
        information into account.

        The returned |Timegrid| object does not depend on the defined
        initialisation period at all:

        >>> from hydpy.core.timetools import TOY
        >>> from hydpy import pub
        >>> pub.timegrids = '2001-10-01', '2010-10-01', '1d'
        >>> TOY.centred_timegrid()[0]
        Timegrid('2000-01-01 12:00:00',
                 '2001-01-01 12:00:00',
                 '1d')

        The same holds for the shape of the returned boolean array:

        >>> len(TOY.centred_timegrid()[1])
        366

        However, the single boolean values depend on whether the respective
        centred date lies at least one time within the initialisation period
        when ignoring the year information.  In our example, all centred dates
        are "relevant" due to the long initialisation period of ten years:

        >>> sum(TOY.centred_timegrid()[1])
        366

        The boolean array contains only the value |True| for all
        initialisation periods covering at least a full year:

        >>> pub.timegrids = '2000-02-01', '2001-02-01', '1d'
        >>> sum(TOY.centred_timegrid()[1])
        366
        >>> pub.timegrids = '2001-10-01', '2002-10-01', '1d'
        >>> sum(TOY.centred_timegrid()[1])
        366

        In all other cases, only the values related to the intersection
        are |True|:

        >>> pub.timegrids = '2001-01-03', '2001-01-05', '1d'
        >>> TOY.centred_timegrid()[1][:5]
        array([False, False,  True,  True, False], dtype=bool)

        >>> pub.timegrids = '2001-12-30', '2002-01-04', '1d'
        >>> TOY.centred_timegrid()[1][:5]
        array([ True,  True,  True, False, False], dtype=bool)
        >>> TOY.centred_timegrid()[1][-5:]
        array([False, False, False,  True,  True], dtype=bool)

        It makes no difference whether initialisation periods not spanning
        a full year contain the 29th of February or not:

        >>> pub.timegrids = '2001-02-27', '2001-03-01', '1d'
        >>> TOY.centred_timegrid()[1][31+28-3-1:31+28+3-1]
        array([False, False,  True,  True,  True, False], dtype=bool)
        >>> pub.timegrids = '2000-02-27', '2000-03-01', '1d'
        >>> TOY.centred_timegrid()[1][31+28-3-1:31+28+3-1]
        array([False, False,  True,  True,  True, False], dtype=bool)
        """
        init = hydpy.pub.timegrids.init
        shift = init.stepsize/2.
        centred = Timegrid(
            cls._STARTDATE+shift,
            cls._ENDDATE+shift,
            init.stepsize)
        if (init.lastdate-init.firstdate) >= '365d':
            return centred, numpy.ones(len(centred), dtype=bool)
        date0 = copy.deepcopy(init.firstdate)
        date1 = copy.deepcopy(init.lastdate)
        date0.year = 2000
        date1.year = 2000
        relevant = numpy.zeros(len(centred), dtype=bool)
        if date0 < date1:
            relevant[centred[date0+shift]:centred[date1+shift]] = True
        else:
            relevant[centred[date0+shift]:] = True
            relevant[:centred[date1+shift]] = True
        return centred, relevant


    def __sub__(self, other: 'TOY') -> float:
        if self >= other:
            return self.seconds_passed - other.seconds_passed
        return self.seconds_passed + other.seconds_left

    def __lt__(self, other: 'TOY') -> bool:
        return self.seconds_passed < other.seconds_passed

    def __le__(self, other: 'TOY') -> bool:
        return self.seconds_passed <= other.seconds_passed

    def __eq__(self, other: Any) -> bool:
        try:
            return self.seconds_passed == other.seconds_passed
        except AttributeError:
            return False

    def __ne__(self, other: Any) -> bool:
        try:
            return self.seconds_passed != other.seconds_passed
        except AttributeError:
            return True

    def __gt__(self, other: 'TOY') -> bool:
        return self.seconds_passed > other.seconds_passed

    def __ge__(self, other: 'TOY') -> bool:
        return self.seconds_passed >= other.seconds_passed

    def __hash__(self):
        return id(str)

    def __str__(self):
        string = '_'.join(str(getattr(self, prop)) for prop
                          in self._PROPERTIES.keys())
        return f"toy_{string}"

    def __repr__(self):
        return "TOY('%s')" % '_'.join(str(getattr(self, prop)) for prop
                                      in self._PROPERTIES.keys())
