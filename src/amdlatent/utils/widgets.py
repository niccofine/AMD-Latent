# PACKAGE IMPORTS

# System packages
import os
from IPython.display import display
import ipywidgets as widgets
from ipyfilechooser import FileChooser

# CUSTOM FUNCTIONS

def checkboxoptions(options, description):
    """
    Function to create a checkbox widget with a description and a list of options. The first option is selected by default.
    Inputs:  - [options: list]: A list of options for the checkbox.
             - [description: str]: A description for the checkbox.
    Output:  - [checkbox]: A checkbox widget with the specified options and description.
    """
    checkbox = widgets.VBox([
        widgets.HTML(value=f"{description}"),
        widgets.HBox(
            [widgets.Checkbox(value=(opt == options[0]), description=opt) for opt in options],
            layout=widgets.Layout(gap='16px')
        )
    ])
    display(checkbox)
    return checkbox.children[1].children

def toggleoptions(options, description):
    """
    Function to create a toggle button widget with a description and a list of options. The first option is selected by default.
    Inputs:  - [options: list]: A list of options for the toggle buttons.
             - [description: str]: A description for the toggle buttons.
    Output:  - [toggle]: A toggle button widget with the specified options and description.
    """
    toggler = widgets.VBox([
        widgets.HTML(value=f"{description}"),
        widgets.ToggleButtons(
            options=options,
            value=options[0]
        )
    ])
    display(toggler)
    return toggler.children[1]

def filechooser(description, dirpath=None):
    """
    Function to create a file chooser widget with a description. The user can select a file from the specified directory or the current working directory if no directory is provided.
    Inputs:  - [description: str]: A description for the file chooser.
             - [dirpath: str, optional]: A path to the directory to start the file chooser in. If None, the current working directory is used.
    Output:  - [filechoose]: A file chooser widget with the specified description and starting directory.
    """
    start_dir = dirpath if dirpath else os.getcwd()
    filechoose = widgets.VBox([
        widgets.HTML(value=f"{description}"),
        FileChooser(start_dir)
    ])
    display(filechoose)
    return filechoose.children[1]

def slideroptions(description, min_val, max_val, step=1, value=None):
    """
    Function to create a slider widget with a description and specified range and step. The slider value can be set to a specific value or defaults to the minimum value.
    Inputs:  - [description: str]: A description for the slider.
             - [min_val: int or float]: The minimum value of the slider.
             - [max_val: int or float]: The maximum value of the slider.
             - [step: int or float, optional]: The step size for the slider. Defaults to 1.
             - [value: int or float, optional]: The initial value of the slider. If None, defaults to min_val.
    Output:  - [slider]: A slider widget with the specified description and range.
    """
    initial_value = value if value is not None else min_val
    slider = widgets.VBox([
        widgets.HTML(value=f"{description}"),
        widgets.FloatSlider(min=min_val, max=max_val, step=step, value=initial_value)
    ])
    display(slider)
    return slider.children[1]

def rangeslideroptions(description, min_val, max_val, step=1, value=None):
    """
    Function to create a range slider widget with a description and specified range and step. The slider values can be set to specific values or defaults to the minimum and maximum values.
    Inputs:  - [description: str]: A description for the range slider.
             - [min_val: int or float]: The minimum value of the range slider.
             - [max_val: int or float]: The maximum value of the range slider.
             - [step: int or float, optional]: The step size for the range slider. Defaults to 1.
             - [value: tuple of (int or float, int or float), optional]: The initial values of the range slider as a tuple (min_value, max_value). If None, defaults to (min_val, max_val).
    Output:  - [range_slider]: A range slider widget with the specified description and range.
    """
    initial_value = value if value is not None else (min_val, max_val)
    range_slider = widgets.VBox([
        widgets.HTML(value=f"{description}"),
        widgets.IntRangeSlider(min=min_val, max=max_val, step=step, value=initial_value)
    ])
    display(range_slider)
    return range_slider.children[1]

def textoptions(description, placeholder=''):
    """
    Function to create a text input widget with a description and an optional placeholder.
    Inputs:  - [description: str]: A description for the text input.
             - [placeholder: str, optional]: A placeholder text for the text input. Defaults to an empty string.
    Output:  - [text_input]: A text input widget with the specified description and placeholder.
    """
    text_input = widgets.VBox([
        widgets.HTML(value=f"{description}"),
        widgets.Textarea(placeholder=placeholder, value =placeholder, layout=widgets.Layout(width='100%', height='100px'))
    ])
    display(text_input)
    return text_input.children[1]