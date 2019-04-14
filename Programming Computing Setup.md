# Using Python on the WRDS Cloud

WRDS Cloud is a high-performance computing cluster to write and execute elaborated research programs. It offers near-line access to all WRDS data and it supports a wide variety of programming languages, including Python. It is composed of two head nodes and a grid of 30 computing nodes. More information is available in the [Introduction to the WRDS Cloud](https://wrds-www.wharton.upenn.edu/pages/support/the-wrds-cloud/introduction-wrds-cloud/). 

To use the Python software and access WRDS data directly on the cloud, it is necessary to connect to the WRDS cloud via *Secure Socket Shell (SSH)*. *SSH* is a remote connection protocol that allows to enter commands and run code in the WRDS native UNIX environment. A *SSH* connection can be established as follows:

```
# Establish a SSH connection to WRDS Cloud (from Terminal):

my-laptop:~ joe$ ssh your_username@wrds-cloud.wharton.upenn.edu
your_username@wrds-cloud.wharton.upenn.edu's password:
[your_username@wrds-cloud-login1-h ~]$
```

> To disconnect from WRDS Cloud, type on the Terminal `logout`

Once connected, it is possible to use Python and access WRDS databases from a command-line, as well as use UNIX commands to interact with the WRDS Cloud. *SSH* establishes a connection to one of the two head nodes of the WRDS Cloud (wrds-cloud-login1 or wrds-cloud-login2). These are designed for high-concurrency traffic and are meant to write programs, examine data and other computationally light activities. 


For computationally intensive activities, it is necessary to establish a connection to the computing nodes, which are designed for for high-performance CPU- and memory-intensive execution. To to do, it is necesary to start an interactive session with *qrsh* as follows:

```
# Start an interactive session (from Terminal):

[your_username@wrds-cloud-login1-h ~]$ qrsh
[your_username@wrds-sas5-h ~]$
```

>The WRDS home directory and a scratch directory are shared across all nodes.

Once the interactive session has been initiated, a *pgpass* needs to be set up. This includes your WRDS username and password and allows to access the WRDS databases without needing to enter the username and password every time a connetion is established. The *pgpass* file can be created by starting the iPython3 shell in an interactive session as follows:

```
# Create a pgpass file (from Terminal):

[your_username@wrds-sas5-h ~]$ ipython3
In [1]: import wrds
In [2]: db = wrds.Connection()
Enter your WRDS username [your_username]:
Enter your password:
In [3]: db.create_pgpass_file()
```

> To disconnect from ipython3, type on the Terminal `quit`

This will require to enter the WRDS username and password only at the first login. Once this file is created, it is sufficient to run the followin code to establish a connection to WRDS:

```
# Establish a connection to WRDS (from Terminal):

db = wrds.Connection()
```

Now the setup is complete and it is possible to start working on jobs. 

## Interactive and Batch Jobs on WRDS Cloud

These are two types of jobs that can be submitted on the WRDS Cloud: interactive jobs, which are executed line-by-line and immediately return a response to each command, like in a Python console; batch jobs, which longer programs executed as a whole, like in a Python run. The former are more useful for exploration and testing, while the latter for elaborated, multi-step programs. Both types of jobs are scheduled and managed by the Grid Engine, which distributed job submissions to the least-busy computing node available. 

To run interactive jobs it is necessary to schedule an interactive job with the WRDS Cloud Grid Engine as follows:

```
# Schedule an interactive job with the Grid Engine (from Terminal):

my-laptop:~ joe$ ssh your_username@wrds-cloud.wharton.upenn.edu
your_username@wrds-cloud.wharton.upenn.edu's password:
[your_username@wrds-cloud-login1-h ~]$ qrsh
[your_username@wrds-sas5-h ~]$ ipython3
In [1]: import wrds
In [2]: db = wrds.Connection()
In [3]: db.raw_sql("SELECT date,dji FROM djones.djdaily")
In [4]: quit
[your_username@wrds-sas6-h ~]$ logout
[your_username@wrds-cloud-login1-h ~]$
```

The above code does creates a *SSH* connection to wrds-cloud.wharton.upenn.edu, submits the job the the Grid Engine which assigns a computing node (in this case number 5), starts an interactive Python session, imports the *wrds* module, initiates a connection to WRDS which uses the *pgpass* file, runs a SQL query.

# Using Python on Your Computer

WRDS provides an interface that allows users to query WRDS data when running Python locally. To access the data, which is stored on a PostgreSQL database, WRDS provides the in-house open-source Python module [wrds](https://github.com/wharton/wrds), which is available on [PyPI](https://pypi.org) and which can be installed as follows:

```
# Install the wrds module (from Terminal):

pip install wrds
```

Once the *wrds* module has been installed, a *pgpass* needs to be set up on the workstation. This includes your WRDS username and password and allows to access the WRDS databases without needing to enter the username and password every time a connetion is established. The *pgpass* file can be created as follows:

```
# Create a pgpass file (from Python Console):

import wrds
db = wrds.Connection(wrds_username='your_username')
db.create_pgpass_file()
```

This will require to enter the WRDS username and password only at the first login. Once this file is created, it is sufficient to run the followin code to establish a connection to WRDS:

```
# Establish a connection to WRDS (from Python Console):

import wrds
db = wrds.Connection(wrds_username='your_username')
```

The connection will be automatically established without needing to re-enter the WRDS username and password.
