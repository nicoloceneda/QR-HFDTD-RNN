# Using Python on the WRDS Cloud

WRDS Cloud is a high-performance computing cluster to write and execute elaborated research programs. It offers near-line access to all WRDS data and it supports a wide variety of programming languages, including Python. More information is available in the [Introduction to the WRDS Cloud](https://wrds-www.wharton.upenn.edu/pages/support/the-wrds-cloud/introduction-wrds-cloud/). 



The WRDS Cloud is comprised of two head nodes and a grid of 30 compute nodes. When you connect to the WRDS Cloud using SSH, you are connecting to one of the head nodes. When you submit a job to run in the WRDS Cloud, you are submitting to one of the available compute nodes for processing.

Head nodes are designed for high-concurrency traffic, while the compute nodes are designed for CPU- and memory-intensive job execution.

Both your WRDS home directory and the scratch directory are shared across all nodes, which allows you to work on any compute node in the grid without needing to transfer any data between systems.




These are two types of jobs that can be submitted on the WRDS Cloud:

Interactive jobs - where commands are run line-by-line and the result for each command is returned as soon as you enter it. Interactive are commonly used to practice with a language, submit simple, smaller queries, or to test expected outcomes before writing a larger program.

Batch jobs - where a whole program is submitted, runs for a length of time, and then returns results to a file. Batch jobs are the most common jobs run in the WRDS Cloud, and are best for larger, multi-step programs that are expected to take hours or days to complete.

Many users use an interactive job to test commands and queries to determine the correct code for a desired output, then create a larger program out of these smaller interactive commands and submit that resulting program as a batch job to the Grid Engine.

# Using Python on Your Computer

WRDS provides an interface that allows users to query WRDS data when running Python locally. To access the data, which is stored on a PostgreSQL database, WRDS provides the in-house open-source Python module [wrds](https://github.com/wharton/wrds), which is available on [PyPI](https://pypi.org) and which can be installed as follows:

```
Install wrds module (from Terminal):

pip install wrds
```

Once the *wrds* module has been installed, a *pgpass* needs to be set up on the workstation. This includes your WRDS username and password and allows to access the WRDS databases without needing to enter the username and password every time a connetion is established. The *pgpass* file can be created as follows:

```
Create a pgpass file (from Python Console):

import wrds
db = wrds.Connection(wrds_username='your_username')
db.create_pgpass_file()
```

This will require to enter the WRDS username and password only at the first login. Once this file is created, to connect to WRDS, it is sufficient to run the followin code:

```
Establish a connection to WRDS (from Python Console):

import wrds
db = wrds.Connection(wrds_username='your_username')
```

The connection will be automatically established without the need to re-enter the WRDS username and password.
