# Using Python on the WRDS Cloud

WRDS Cloud is a high-performance computing cluster to write and execute elaborated research programs. It offers near-line access to all WRDS data and it supports a wide variety of programming languages, including Python. It is composed of two head nodes and a grid of 30 computing nodes. More information is available in the [Introduction to the WRDS Cloud](https://wrds-www.wharton.upenn.edu/pages/support/the-wrds-cloud/introduction-wrds-cloud/). 

To use the Python software and access WRDS data directly on the cloud, it is necessary to connect to the WRDS cloud via *Secure Socket Shell* (SSH). *SSH* is a remote connection protocol that allows to enter commands and run code in the WRDS native UNIX environment. A *SSH* connection can be established as follows:

```
# Establish a SSH connection to WRDS Cloud (from Terminal):

my-laptop:~ joe$ ssh your_username@wrds-cloud.wharton.upenn.edu
your_username@wrds-cloud.wharton.upenn.edu's password:
[your_username@wrds-cloud-login1-h ~]$
```

Once connected, it is possible to use Python and access WRDS databases from a command-line, as well as use UNIX commands to interact with the WRDS Cloud. *SSH* establishes a connection to one of the two head nodes of the WRDS Cloud (wrds-cloud-login1 or wrds-cloud-login2). These are designed for high-concurrency traffic and are meant to write programs, examine data and other computationally light activities. 
> To disconnect from WRDS Cloud, type on the Terminal `logout`

For computationally intensive activities, it is necessary to establish a connection to the computing nodes, which are designed for for high-performance CPU- and memory-intensive execution. 

```
```

The WRDS home directory and a scratch directory are shared across all nodes.




These are two types of jobs that can be submitted on the WRDS Cloud:

Interactive jobs - where commands are run line-by-line and the result for each command is returned as soon as you enter it. Interactive are commonly used to practice with a language, submit simple, smaller queries, or to test expected outcomes before writing a larger program.

Batch jobs - where a whole program is submitted, runs for a length of time, and then returns results to a file. Batch jobs are the most common jobs run in the WRDS Cloud, and are best for larger, multi-step programs that are expected to take hours or days to complete.

Many users use an interactive job to test commands and queries to determine the correct code for a desired output, then create a larger program out of these smaller interactive commands and submit that resulting program as a batch job to the Grid Engine.

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
