import re


def filter_word(raw_word):
    if not re.search(r'^[\u4e00-\u9fa5_a-zA-Z0-9]+$', raw_word):
        return False
    else:
        return True


def filter_char(char):
    if not re.search(r'[\u4e00-\u9fa5_a-zA-Z0-9]', char):
        return False
    else:
        return True
