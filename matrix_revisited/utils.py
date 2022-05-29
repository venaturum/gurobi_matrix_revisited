import numpy as np
import pandas as pd
from IPython.display import display_html


def run(f):
    """Used as a decorator to assign run function in place and assign output"""
    return f()


def display_matrices(*args):
    html_str = ""
    padding = pd.Series([""] * len(df))
    for df in args:
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += (
            pd.concat([padding, pd.DataFrame(df), padding], axis=1)
            .to_html(index=False, header=False)
            .replace("table", 'table style="float:left"')
        )
        html_str += "</td></th>"
    display_html(html_str, raw=True)


def get_parameters_from_function(func):
    return func.__code__.co_varnames[: func.__code__.co_argcount]
