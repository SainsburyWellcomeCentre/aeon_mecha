# Local set-up using pip and venv

1) Ensure that you have python 3.9 and pip installed on your computer:
```
python -V         # should return >=3.9
python -m pip -V
```

2) Install virtualenv: `python -m pip install virtualenv`

3) Create virtual environment: `python -m venv aeon`

4) Activate the virtual environment and install the code dependencies:
```
source aeon/bin/activate
python -m pip install -r requirements.txt
```

5) Using the virtual environment:

`source aeon/bin/activate` activates the virtual environment.

`deactivate` deactivates the virtual environment.