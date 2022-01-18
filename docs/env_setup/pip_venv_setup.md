## Local set-up using pip and venv

It is assumed that you already have python 3.10 and pip installed on your computer.

On Windows:

1) Install virtualenv:

`python -m pip install virtualenv`

2) Create virtual environment:

`python -m venv aeon`

3) Activate the virtual environment and install the code dependencies:

```
.\aeon\Scripts\activate
python -m pip install -r requirements.txt
```

4) Using the virtual environment:

`.\aeon\Scripts\activate` activates the virtual environment.

`deactivate` deactivates the virtual environment.

On MacOS and GNU/Linux:

1) Install virtualenv:

`python3 -m pip install virtualenv`

2) Create virtual environment:

`python3 -m venv aeon`

3) Activate the virtual environment and install the code dependencies:

`source aeon/bin/activate`

`python3 -m pip install -r requirements.txt`

4) Using the virtual environment:

`source aeon/bin/activate` activates the virtual environment.

`deactivate` deactivates the virtual environment.