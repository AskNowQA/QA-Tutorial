The code exists in the form of Python (ver 3.6) notebooks.
The following instructions are written for linux based systems, and would need to be adapted for other OS.

## Local Python Environment
We begin the setup by setting up a local environment with python3.6.

If you already use conda/virtualenv/venv based environments, feel free to skip ahead.

**What is a virtual environment?** "A self-contained directory tree that contains a Python installation for a particular version of Python, plus a number of additional packages." [[1](https://docs.python.org/3/tutorial/venv.html)]


1. Update `pip` (if you haven't, in a while). 
For Ubuntu based systems you could simply use `sudo pip3 install --upgrade pip` 

2. Install virtualenv by `pip install --user virtualenv`

3. Check if virtualenv works correctly. Simply run `virtualenv` on your terminal and see if the command is recognized.
	**If not**, your local path probably isn't included in `$PATH`. 
	Simply run `PATH="$PATH:$HOME/.local/bin"`, 
		and preferably save it in your configuration file for persistency (e.g. in `~/.bashrc`). 

4. Create a virtual environment with python3.6. `virtualenv <ENVDIR>/<ENVNAME> --python=python3.6` where `ENVDIR` and `ENVNAME` are something of your choosing. For instance, my `ENVDIR` would be `~/Dev/venv` and `ENVNAME` would be `newenv`

5. Activate the env by `source <ENVDIR>/<ENVNAME>/bin/activate`
	You should now see a prefix on your prompt.
	For instance, when my prompt is `tutorial@priyansh-fhg:~$` before the activation, it turns to `<ENVNAME> tutorial@priyansh-fhg:~$`

	To deactivate it, simply type `deactivate` on the terminal.

We can now proceed to configure this environment to our needs.

## Installing packages

1. Clone the repo, and navigate to it on the terminal. 

2. Change the branch to this one by `git checkout 2019_tutorial`

3. Make the setup shell script executable. `chmod +x setup.sh`

4. Install the requirements, and some needed resources (like Word Embeddings etc). Simply run the setup script - `./setup.sh`
	This will take _a while_.

5. Install a version of pytorch depending on your system. Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to find what works for you.
NOTE: We need pytorch 0.4 version, and finding it is going to take a moment. 

6. Download a pre-processed dataset file from Google Drive, and place it in `resources` directory within the cloned repo. Visit [https://drive.google.com/open?id=1AvBIC2QXmJ9tVzU4blAh2uY6canug6j9](https://drive.google.com/open?id=1AvBIC2QXmJ9tVzU4blAh2uY6canug6j9).

And we're good to go.
