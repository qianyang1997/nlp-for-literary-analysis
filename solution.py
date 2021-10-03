import re


def find_function_names(text: str) -> list:
    """
    This function scrapes function names from python 2 scripts formatted as text.

    Params:
        text: str - python 2 script
    Returns:
        list of str of function names, if any
    """
    result = re.findall(r'def\s(\w+)\(.*\):\s+', text)
    return result


def find_dates(text: str) -> list:
    """
    This function scrapes all dates from text as formatted below:
        1) MM/dd/YYYY or M/d/YYYY or M/dd/YYYY or MM/d/YYYY
        2) YYYY/MM/dd or YYYY/M/d or YYYY/MM/d or YYYY/M/dd
        3) Month D, Yr
        4) MM-dd-YYYY or M-d-YYYY or M-dd-YYYY or MM-d-YYYY
        5) Mon(abbr.) D, Yr

    Params:
        text: str - input text to scrape dates from
    Returns:
        list of tuples in the format of (month, day, year)
    """
    patterns = [
        r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
        r'\b(\d{4})/(\d{1,2})/(\d{1,2})\b',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s(\d{1,2}),\s(\d{4})\b',
        r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b',
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(\d{1,2}),\s(\d{4})\b'
    ]

    dates = list()

    for i, pattern in enumerate(patterns):
        match = re.findall(pattern, text)
        for result in match:
            if i == 0 or i == 3:
                month = int(result[0])
                day = int(result[1])
                year = int(result[2])
            elif i == 1:
                month = int(result[1])
                day = int(result[2])
                year = int(result[0])
            else:
                month = result[0]
                day = int(result[1])
                year = int(result[2])

            if (type(month) == str or (1 <= month <= 12)) and (1 <= day <= 31):
                dates += [(month, day, year)]

    return dates


if __name__ == '__main__':
    snippet = """
    def import(x='): '):
        something
        something
    
    date = 'Feb 02, 2020'
    """
    print(find_function_names(snippet))
    print(find_dates(snippet))
