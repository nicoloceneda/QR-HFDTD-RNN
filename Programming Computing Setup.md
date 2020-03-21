# 1 - Using Python on the WRDS Cloud

WRDS Cloud is a high-performance computing cluster to write and execute elaborated programs. It supports a wide variety of programming languages, including Python, and it offers access to all WRDS data. It is composed of 2 head nodes and a grid of 30 computing nodes. More information is available in the [Introduction to the WRDS Cloud](https://wrds-www.wharton.upenn.edu/pages/support/the-wrds-cloud/introduction-wrds-cloud/). 

To use the Python software and access WRDS data directly on the cloud, it is necessary to connect to the WRDS cloud via *Secure Socket Shell (SSH)*, which is a remote connection protocol that allows to use Python and access WRDS databases from a command-line, as well as to use *UNIX* commands to interact with the WRDS Cloud. It establishes a connection to one of the 2 head nodes of the WRDS Cloud (wrds-cloud-login1 or wrds-cloud-login2). These are designed for high-concurrency traffic and are meant to write programs, examine data and other computationally light activities. A `SSH` connection can be established as follows:

```
# Establish a SSH connection to WRDS Cloud (from Terminal):

my_laptop:my_directory my_name$ ssh my_wrds_username@wrds-cloud.wharton.upenn.edu
my_wrds_username@wrds-cloud.wharton.upenn.edu's password:
[my_wrds_username@wrds-cloud-login1-w ~]$
```

> To disconnect from the head node (i.e. from the WRDS Cloud), type `logout` in the Terminal.

For computationally intensive activities, it is necessary to establish a connection to the computing nodes, which are designed for high-performance CPU- and memory-intensive execution. To to do, it is necesary to start an interactive session with `qrsh` as follows:

```
# Start an interactive session (from Terminal):

[my_wrds_username@wrds-cloud-login1-w ~]$ qrsh
[my_wrds_username@wrds-sas5-w ~]$
```

> To disconnect from the computing node node (and go back to the head node), type `logout` in the Terminal.

>The WRDS home directory and a scratch directory are shared across all nodes.

Once the interactive session has been initiated, a *pgpass* needs to be set up. This includes your WRDS username and password and allows to access the WRDS databases, without needing to enter the username and password every time a connetion is established. The *pgpass* file can be created by starting the `iPython3` shell in an interactive session as follows:

```
# Create a pgpass file (from Terminal):

[my_wrds_username@wrds-sas5-h ~]$ ipython3
In [1]: import wrds
In [2]: db = wrds.Connection()
Enter your WRDS username [my_wrds_username]:
Enter your password:
In [3]: db.create_pgpass_file()
```

> To disconnect from ipython3, type `quit` in the Terminal.

This will require to enter the WRDS username and password only at the first login. Once this file is created, it is sufficient to run the following code to establish a connection to WRDS:

```
# Establish a connection to WRDS (from Terminal):

[my_wrds_username@wrds-sas5-h ~]$ ipython3
In [1]: import wrds
In [2]: db = wrds.Connection()
```

Now the setup is complete and it is possible to start working on jobs. There are two types of jobs that can be submitted on the WRDS Cloud: interactive jobs, which are executed line-by-line and immediately return a response to each command, like in a Python console; batch jobs, which are longer programs executed as a whole, like in a Python run. The former are more useful for exploration and testing, while the latter are more useful for elaborated, multi-step programs. Both types of jobs are scheduled and managed by the Grid Engine, which is a distributed resource management system that optimizes job execution for large high-performance computing clusters by distributing job submissions to the least-busy computing node. 

## 1.1 - Interactive Jobs

To run interactive jobs, it is necessary to schedule an interactive job with the WRDS Cloud Grid Engine. As with all jobs on the WRDS Cloud, interactive jobs are submitted from one of the head nodes and run on one of the computing nodes. 

```
# Schedule an interactive job with the Grid Engine (from Terminal):

my_laptop:my_directory my_name$ ssh my_wrds_username@wrds-cloud.wharton.upenn.edu
my_wrds_username@wrds-cloud.wharton.upenn.edu's password:
[my_wrds_username@wrds-cloud-login1-h ~]$ qrsh
[my_wrds_username@wrds-sas5-h ~]$ ipython3
In [1]: import wrds
In [2]: db = wrds.Connection()

------- Up to here is the setup explained in the previous section -------

In [3]: db.raw_sql("select time_m, size, price from taqmsec.ctm_20180102 where sym_root = 'AAPL' ")
In [4]: quit
[my_wrds_username@wrds-sas5-h ~]$ logout
[my_wrds_username@wrds-cloud-login1-h ~]$
```

The above code creates a `SSH` connection to wrds-cloud.wharton.upenn.edu, submits the interactive job to the Grid Engine, which assigns a computing node (in this case number 5), starts an interactive Python session, imports the `wrds` module, initiates a connection to WRDS, which uses the crdentials in the *pgpass* file, and runs a SQL query.

## 1.2 - Batch Jobs

To run batch jobs two files are needed: a Python program (.py) to be executed and a wrapper shell script (.sh) to be submitted to the Grid Engine to specify the software to use and the program to run. It is this wrapper script that is submitted to Grid Engine, not the research program. More precisely, as with all jobs on the WRDS Cloud, batch jobs are submitted from one of the head nodes and run on one of the computing nodes. 

> Note that a batch job requires a *pgpass* file as it cannot prompt for passwords.

The first step is the creation of the Python program `my_program.py`, using a command line editor such as *nano* or writing it on your local computer and uploading it via *SFTP*. Here is an example of the content of this program:

```
# Python program script (.py file):

import wrds
db = wrds.Connection()
data = db.raw_sql("select time_m, size, price from taqmsec.ctm_20180102")
data.to_csv("ctm_20180102.csv")
```

The above code establishes a connection to WRDS, runs a SQL query and outputs the result to a .csv file.

The second step is the creation of the wrapper shell script `my_program.sh`, using a command line editor such as *nano* or writing it on your local computer and uploading it via *SFTP*. Here is an example of the content of this program:

```
# Wrapper shell script (.sh file):

#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M my_email_address
python3 my_program.py
```

> Note that since the wrapper script is simply a shell script, it supports all UNIX commands. 

The above code sets the shell of the wrapper script to `bash`, instructs (with `cwd`) the Grid Engine to look into the current directory for referenced files and to store the output in the same directory, sends an email to the specified address when the job starts and terminates, and runs the program `my_program.py` using Python 3. 

Now that both files have been created, the wrapper script can be submitted using the `qsub` command, as follows:

```
# Submit the batch job (from Terminal):

my_laptop:my_directory my_name$ ssh my_wrds_username@wrds-cloud.wharton.upenn.edu
my_wrds_username@wrds-cloud.wharton.upenn.edu's password:
[my_wrds_username@wrds-cloud-login1-h ~]$ qsub my_program.sh
```

The Grid Engine will then run the batch job and return several output files to the same directory of the wrapper script (as instructed by `#$ -cwd`): a my_program.csv file, which is the output of the Python program; a my_program.sh.o##### file, which is the Grid Engine file that contains all the output from the my_program.sh file; a my_program.sh.e##### file, which contains all the errors of the my_program.sh file. ##### stands for the Grid Engine job number.

> Note that running the program multiple times will overwrite the my_program.csv file with the new output. To avoid this, it is sufficient to rename the initial output with the command `mv my_program.csv new_name.csv`. On the contrary, the Grid Engine output and error files are not overwritten as their name contains the job number, which makes them unique.

### 1.2.1 - Transferring Files with SFTP

The easiest way to run batch jobs is to create the Python program (.py) and the wrapper shell script (.sh) locally on your computer and then transfer them to the WRDS Cloud via *Secure FTP (SFTP)*. This is a remote filesystem browser that allows to manage files across directories on remote servers and to download and upload data betweem the remote server and the local workstation. This is a convenient way to connect to the WRDS Cloud to manage the files contained in the personal home directory and the scratch directory, browse through the WRDS data and upload or download anything. 

A suggested SFTS browser for Mac is *CyberDuck*, which can be downloaded [here](https://cyberduck.io). Once the SFTP client has been opened on the local workstation, to connect to WRDS Cloud it is necessary to create a new connection with the following parameters:
* Server: wrds-cloud.wharton.upenn.edu
* Port: 22
* Username: Your WRDS Username
* Password: Your WRDS Password

Once this is done, it is sufficient to click on Connect to be redirected to your WRDS Cloud home directory `/home/institution/user`. It is now possible to drag and drop files between the local workstation and the WRDS Cloud, as well as changing directory.

The WRDS data are located in the `/wrds` directory. A list of all WRDS datasets and their file system locations is available [here](https://wrds-web.wharton.upenn.edu/wrds/tools/variable.cfm?_ga=2.114595075.561933824.1556371438-601882553.1555849734)

## 1.3 - Installing Python Packages

Although WRDS Cloud comes with many pre-installed Python packages, it also allows to install new packages (or different versions of the pre-installed ones) to make them available for interactive and batch jobs. 

```
# Display the list of pre-installed Python packages (from Terminal):

my_laptop:my_directory my_name$ ssh my_wrds_username@wrds-cloud.wharton.upenn.edu
my_wrds_username@wrds-cloud.wharton.upenn.edu's password:
[my_wrds_username@wrds-cloud-login1-h ~]$ pip3 list
```

To install a new package on the WRDS Cloud it is necessary to execute three sequential commands to create a *virtualenv* in the WRDS Cloud home directory from one of the head nodes, activate the *virtualenv*, and download the package needed using `pip`. Note that since the computing nodes are not internet-accessible, it is necessary to use the two head nodes to upload packages to the WRDS Cloud home directory. However, once uploaded, these packages can also be used on the computing nodes.

```
# Install a Python package (from Terminal):

my_laptop:my_directory my_name$ ssh my_wrds_username@wrds-cloud.wharton.upenn.edu
my_wrds_username@wrds-cloud.wharton.upenn.edu's password:
[my_wrds_username@wrds-cloud-login1-h ~]$ virtualenv3 --system-site-packages ~/virtualenv
[my_wrds_username@wrds-cloud-login1-h ~]$ source ~/virtualenv/bin/activate
(virtualenv) [my_wrds_username@wrds-cloud-login1-h ~]$ pip3 install your_package
```

### 1.3.1 - Interactive Jobs

Each time you want to use the newly installed package in an interactive job, it is sufficient to start an interactive session from a WRDS Cloud head node, activate the *virtualenv* created in the section above, and then start the interactive job. 

```
# Activate the virtualenv (from Terminal):

[my_wrds_username@wrds-cloud-login1-h ~]$ qrsh
[my_wrds_username@wrds-sas5-h ~]$ source ~/virtualenv/bin/activate
(virtualenv) [my_wrds_username@wrds-sas5-h ~]$ ipython3
In [1]: import your_package
```
> To disconnect from ipython3, type `quit`; to deactivate the *virtualenv* type `deactivate`; to disconnect from the interactive session type `logout`.

### 1.3.2 - Batch Jobs

Instead, to use the newly installed package in a batch job, it is sufficient to write the Python script as usual (making use of the package needed), include a line to activate the *virtualenv* in the wrapper script (before Python is called), and submit the wrapper script. The following code illustrates how to modify the wrapper sript. 

```
# Activate the virtualenv in the wrapper shell script (.sh file):

#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M my_email_address
source ~/virtualenv/bin/activate
python3 my_program.py
```

### 1.4 - Monitoring and Managing Jobs

Once an interactive or batch job has been submitted to the WRDS Cloud, it is usually necessary to monitor and manage its status. The following commands allow to do so:

* `qstat`: Shows all running and queued jobs submitted by you. If no result is returned, then it means that all jobs have been completed. 
* `qstat -u \*`: Shows all running and queued jobs submitted by all users.
* `qstat -j 1234567`: Shows detailed status information about the running job 1234567.
* `qstat -f`: Shows all available queues and their status (d = disabled for maintenance, a = fully utilized by jobs).
* `qhost`: Show all computing nodes, including number of processors and amount of RAM per node.
* `qhost -j`: Same as above, but also shows jobs per node.
* `qdel 1234567`: Delete job 1234567

More details about the options provided by the Grid Engine are available [here](http://www.univa.com/resources/files/univa_user_guide_univa__grid_engine_854.pdf); more details about monitoring and managing jobs are available [here](https://wrds-www.wharton.upenn.edu/pages/support/the-wrds-cloud/running-jobs/managing-jobs-wrds-cloud/).

# 2 - Using Python on Your Computer

WRDS provides an interface that allows users to query WRDS data when running Python locally. To access the data, which is stored on a PostgreSQL database, WRDS provides the in-house open-source Python module [wrds](https://github.com/wharton/wrds), which is available on [PyPI](https://pypi.org) and which can be installed as follows:

```
# Install the wrds module (from Terminal):

my_laptop:my_directory my_name$ pip install wrds
```

Once the `wrds` module has been installed, a *pgpass* needs to be set up on the workstation. This includes your WRDS username and password and allows to access the WRDS databases without needing to enter the username and password every time a connetion is established. The *pgpass* file can be created as follows:

```
# Create a pgpass file (from Python Console):

In [1]: import wrds
In [2]: db = wrds.Connection(wrds_username='my_wrds_username')
In [3]: db.create_pgpass_file()
```

This will require to enter the WRDS username and password only at the first login. Once this file is created, it is sufficient to run the following code to establish a connection to WRDS:

```
# Establish a connection to WRDS (from Python Console):

In [1]: import wrds
In [2]: db = wrds.Connection(wrds_username='my_wrds_username')
```
