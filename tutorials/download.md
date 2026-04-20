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
[mkandes@login02 ~]$ wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
--2026-04-20 09:29:27--  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
Resolving www.cs.toronto.edu (www.cs.toronto.edu)... 128.100.3.30
Connecting to www.cs.toronto.edu (www.cs.toronto.edu)|128.100.3.30|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 170498071 (163M) [application/x-gzip]
Saving to: ‘cifar-10-python.tar.gz’

cifar-10-python.tar.gz           100%[==========================================================>] 162.60M  31.9MB/s    in 5.4s    

2026-04-20 09:29:33 (30.1 MB/s) - ‘cifar-10-python.tar.gz’ saved [170498071/170498071]

[mkandes@login02 ~]$
```

After the download completes, go ahead and list the files in your HOME directory using the [`ls`](https://en.wikipedia.org/wiki/Ls) command to check out how much data we've downloaded.

*Command*
```
ls -lh
```

*Output*
```
[mkandes@login02 ~]$ ls -lh
total 3.3G
-rw-r--r--   1 mkandes use300 163M Jun  4  2009 cifar-10-python.tar.gz
...
[mkandes@login02 ~]$ 
```

The dataset we've downloaded has been delieved to us as a [`gzip`](https://en.wikipedia.org/wiki/Gzip)-compressed `tar` file. To extract the dataset from the "tarball", use the [`tar`](https://en.wikipedia.org/wiki/Tar_(computing)) command.

```
```
