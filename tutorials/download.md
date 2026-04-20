# CIFAR through the tubes: Downloading data from the internet

<img src='../images/usdci2025.jpg' width='100%' height='100%'/>

[Image Credit: B. J. Roberts, NLR](https://research-hub.nlr.gov/en/publications/data-center-infrastructure-in-the-united-states-november-2025-2/)

The aim of this tutorial is to introduce you to command-line tools that are useful for downloading data from the internet and verifing the data is correct. The dataset we'll be working with is the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), a well-known machine learning dataset that consists of 60K 32x32 colour images broken out into 10 classes, with 6000 images per class.

Let's get started by logging into Expanse with your account either directly via `ssh` and the [Expanse User Portal](https://portal.expanse.sdsc.edu)

*Command*
```
ssh username@login.expanse.sdsc.edu
```

*Output*
```
mkandes@hardtack:~$ ssh mkandes@login.expanse.sdsc.edu
(mkandes@login.expanse.sdsc.edu) TOTP code for mkandes: XXXXXX
Welcome to Bright release         9.0

                                                         Based on Rocky Linux 8
                                                                    ID: #000002

--------------------------------------------------------------------------------

                                 WELCOME TO
                  _______  __ ____  ___    _   _______ ______
                 / ____/ |/ // __ \/   |  / | / / ___// ____/
                / __/  |   // /_/ / /| | /  |/ /\__ \/ __/
               / /___ /   |/ ____/ ___ |/ /|  /___/ / /___
              /_____//_/|_/_/   /_/  |_/_/ |_//____/_____/

--------------------------------------------------------------------------------

Use the following commands to adjust your environment:

'module avail'            - show available modules
'module add <module>'     - adds a module to your environment for this session
'module initadd <module>' - configure module to be loaded at every login

-------------------------------------------------------------------------------
Last login: Sun Apr 19 15:30:14 2026 from 216.15.51.171
[mkandes@login02 ~]$
```

Once you are logged into Expanse, please go ahead and download the CIFAR-10 dataset using [wget](https://www.gnu.org/software/wget), a command-line program for retrieving files via HTTP, HTTPS, FTP and FTPS protocols.

*Command*  
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

*Output*
```
[mkandes_test@login02]~% wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
--2026-04-20 09:42:07--  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
Resolving www.cs.toronto.edu (www.cs.toronto.edu)... 128.100.3.30
Connecting to www.cs.toronto.edu (www.cs.toronto.edu)|128.100.3.30|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 170498071 (163M) [application/x-gzip]
Saving to: ‘cifar-10-python.tar.gz’

cifar-10-python.tar.gz           100%[==========================================================>] 162.60M  36.1MB/s    in 5.0s    

2026-04-20 09:42:13 (32.3 MB/s) - ‘cifar-10-python.tar.gz’ saved [170498071/170498071]

[mkandes_test@login02]~%
```

After the download completes, go ahead and list the files in your HOME directory using the [`ls`](https://en.wikipedia.org/wiki/Ls) command to check out how much data we've downloaded.

*Command*
```
ls -lh
```

*Output*
```
[mkandes_test@login02]~% ls -lh
total 163M
-rw-r--r-- 1 mkandes_test ddp386 163M Jun  4  2009 cifar-10-python.tar.gz
[mkandes_test@login02]~%
```

The dataset we've downloaded has been delieved to us as a [`gzip`](https://en.wikipedia.org/wiki/Gzip)-compressed `tar` file. To extract the dataset from the "tarball", use the [`tar`](https://en.wikipedia.org/wiki/Tar_(computing)) command.

*Command*
```
tar -xf cifar-10-python.tar.gz
```

*Output*
```
[mkandes_test@login02]~% tar -xf cifar-10-python.tar.gz
[mkandes_test@login02]~% ls -lh
total 163M
drwxr-xr-x 2 mkandes_test ddp386   10 Jun  4  2009 cifar-10-batches-py
-rw-r--r-- 1 mkandes_test ddp386 163M Jun  4  2009 cifar-10-python.tar.gz
[mkandes_test@login02]~%
```

With the data extracted from the tarball to a directory, let's check out what's inside.

*Command*
```
ls -lh cifar-10-batches-py/
```

*Output*
```
[mkandes_test@login02]~% ls -lh cifar-10-batches-py/
total 177M
-rw-r--r-- 1 mkandes_test ddp386 158 Mar 30  2009 batches.meta
-rw-r--r-- 1 mkandes_test ddp386 30M Mar 30  2009 data_batch_1
-rw-r--r-- 1 mkandes_test ddp386 30M Mar 30  2009 data_batch_2
-rw-r--r-- 1 mkandes_test ddp386 30M Mar 30  2009 data_batch_3
-rw-r--r-- 1 mkandes_test ddp386 30M Mar 30  2009 data_batch_4
-rw-r--r-- 1 mkandes_test ddp386 30M Mar 30  2009 data_batch_5
-rw-r--r-- 1 mkandes_test ddp386  88 Jun  4  2009 readme.html
-rw-r--r-- 1 mkandes_test ddp386 30M Mar 30  2009 test_batch
[mkandes_test@login02]~%
```

What type of files are these? Let's check the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) website again. See [Pickle](https://en.wikipedia.org/wiki/Serialization#Pickle). 
