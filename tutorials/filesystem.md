# More files, more problems: Advantages and limitations of different filesystems

<img src='https://wiki.lustre.org/images/a/a3/Lustre_File_System_Overview_%28DNE%29_lowres_v1.png' width='100%' height='100%'/>

[Image Credit: Malcom, wiki.lustre.org](https://wiki.lustre.org)

The aim of this tutorial is to teach you about the advantages and limitations of different [filesystems](https://en.wikipedia.org/wiki/File_system) that you'll typically find available on a high-performance computing system.

## Login and Navigate to Your Working Directory

If you are not already logged in to Expanse, please login to your account either directly via `ssh` or through the [Expanse User Portal](https://portal.expanse.sdsc.edu) and navigate to your working directory for the tutorial exercises.

*Command*
```
ssh <username>@login.expanse.sdsc.edu
```

*Output*
```
mkandes@hardtack:~$ ssh mkandes@login.expanse.sdsc.edu
(mkandes@login.expanse.sdsc.edu) TOTP code for mkandes: 522652
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
Last login: Mon Apr 20 12:49:26 2026 from 216.15.51.171
[mkandes@login02 ~]$
```

*Command*
```
cd complecs/
```

*Output*
```
[mkandes@login02 ~]$ cd complecs/
[mkandes@login02 complecs]$ pwd
/home/mkandes/complecs
[mkandes@login02 complecs]$ ls -lahtr
total 164M
drwxr-xr-x  2 mkandes use300   10 Jun  4  2009 cifar-10-batches-py
-rw-r--r--  1 mkandes use300 163M Jun  4  2009 cifar-10-python.tar.gz
drwxr-x--- 29 mkandes use300   46 Apr 20 15:28 ..
-rw-r--r--  1 mkandes use300   57 Apr 20 15:59 cifar-10-python.tar.gz.md5
drwxr-xr-x  3 mkandes use300    6 Apr 20 16:04 .
-rw-r--r--  1 mkandes use300   89 Apr 20 16:04 cifar-10-python.tar.gz.sha256
[mkandes@login02 complecs]$
```

## Clone the Dataset to Your Working Directory

Once logged in, go ahead and try to clone this [GitHub repository](https://github.com/YoongiKim/CIFAR-10-images.git) that contains a copy of the CIFAR-10 dataset to your working directory. Note, however, please be prepared to **cancel** the download. 

*Command*
```
git clone https://github.com/YoongiKim/CIFAR-10-images.git
```

*Output*
```
[mkandes@login02 complecs]$ git clone https://github.com/YoongiKim/CIFAR-10-images.git
Cloning into 'CIFAR-10-images'...
remote: Enumerating objects: 60027, done.
remote: Total 60027 (delta 0), reused 0 (delta 0), pack-reused 60027 (from 1)
Receiving objects: 100% (60027/60027), 19.94 MiB | 38.24 MiB/s, done.
Resolving deltas: 100% (59990/59990), done.
Updating files:  15% (9003/60001)
```

If you have not done so already, please **cancel** your `git clone` command.

*Command*
```
Ctrl+C
```

*Output*
```
[mkandes@login02 complecs]$ git clone https://github.com/YoongiKim/CIFAR-10-images.git
Cloning into 'CIFAR-10-images'...
remote: Enumerating objects: 60027, done.
remote: Total 60027 (delta 0), reused 0 (delta 0), pack-reused 60027 (from 1)
Receiving objects: 100% (60027/60027), 19.94 MiB | 38.24 MiB/s, done.
Resolving deltas: 100% (59990/59990), done.
^Cwarning: Clone succeeded, but checkout failed.
You can inspect what was checked out with 'git status'
and retry with 'git restore --source=HEAD :/'


[mkandes@login02 complecs]$
```

Unfortunately, it'll take far too long for all of us to download this version of the dataset to our home directories. How long?  Well, here is one measurement using the [`time`](https://en.wikipedia.org/wiki/Time_(Unix)) command.

*Command*
```
time -p git clone https://github.com/YoongiKim/CIFAR-10-images.git
```

*Output*
```
[mkandes@login02 complecs]$ time -p git clone https://github.com/YoongiKim/CIFAR-10-images.git
Cloning into 'CIFAR-10-images'...
remote: Enumerating objects: 60027, done.
remote: Total 60027 (delta 0), reused 0 (delta 0), pack-reused 60027 (from 1)
Receiving objects: 100% (60027/60027), 19.94 MiB | 30.39 MiB/s, done.
Resolving deltas: 100% (59990/59990), done.
Updating files: 100% (60001/60001), done.
real 2222.33
user 1.73
sys 5.69
[mkandes@login02 complecs]$
```

Why does it take so much time?

## Clone the Dataset to Your Scratch Directory

Before we answer the question above, let's try to clone the dataset to an alternative location on Expanse. To make your life simpler, define the following [`alias`](https://en.wikipedia.org/wiki/Alias_(command)) command to start an interactive session on one of Expanse's compute nodes. 

*Command*
```
alias start-interactive='srun --partition=debug --account=sdp157 --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=4G --time=00:30:00 --pty --wait=0 /bin/bash'
```

*Output*
```
[mkandes@login02 complecs]$ alias start-interactive='srun --partition=debug --account=sdp157 --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=4G --time=00:30:00 --pty --wait=0 /bin/bash'
[mkandes@login02 complecs]$
```
