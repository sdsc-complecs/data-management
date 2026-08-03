# More files, more problems: Advantages and limitations of different filesystems

<img src='https://wiki.lustre.org/images/a/a3/Lustre_File_System_Overview_%28DNE%29_lowres_v1.png' width='100%' height='100%'/>

[Image Credit: Malcom, wiki.lustre.org](https://wiki.lustre.org)

The aim of this tutorial is to teach you about some of the advantages and limitations of different [filesystems](https://en.wikipedia.org/wiki/File_system) that you'll typically find available on a high-performance computing system.

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

Before we answer the question above, let's try to clone the dataset to an alternative location on Expanse. Start an interactive session on one of Expanse's compute nodes. 

*Command*
```
srun --partition=shared --account=sdp157 --reservation=si26cpu --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=4G --time=00:30:00 --pty --wait=0 /bin/bash
```

*Output*
```
[mkandes@login01 ~]$ srun --partition=shared --account=sdp157 --reservation=si26cpu --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=4G --time=00:30:00 --pty --wait=0 /bin/bash
[mkandes@exp-1-23 ~]$
```


Once your interactive session starts, navigate to your `/scratch` working directory. 

*Command*
```
cd "/scratch/${USER}/job_${SLURM_JOB_ID}"
```

*Output*
```
[mkandes@exp-1-23 ~]$ cd "/scratch/${USER}/job_${SLURM_JOB_ID}"
[mkandes@exp-1-23 job_52929778]$ ls -lahtr
total 8.0K
drwxr-xr-x 3 root    root 4.0K Aug  3 15:46 ..
drwx------ 2 mkandes root 4.0K Aug  3 15:46 .
[mkandes@exp-1-23 job_52929778]$
```

Now, try and clone the dataset here. What do you observe?

*Command*
```
time -p git clone https://github.com/YoongiKim/CIFAR-10-images.git
```

*Output*
```
[mkandes@exp-1-23 job_52929778]$ time -p git clone https://github.com/YoongiKim/CIFAR-10-images.git
Cloning into 'CIFAR-10-images'...
remote: Enumerating objects: 60027, done.
remote: Total 60027 (delta 0), reused 0 (delta 0), pack-reused 60027 (from 1)
Receiving objects: 100% (60027/60027), 19.94 MiB | 44.49 MiB/s, done.
Resolving deltas: 100% (59990/59990), done.
real 2.36
user 0.66
sys 1.04
[mkandes@exp-1-23 job_52929778]$
```

Why is there such a large difference in the time to download the same dataset when the only thing we've changed is the target directory? Hint: What type of underlying filesystems are in use? You can check basic filesystem information with the [`df`](https://en.wikipedia.org/wiki/Df_(Unix)) command. Start by running the `df` command on the `/scratch` directory.

*Command*
```
df -Th /scratch
```

*Output*
```
[mkandes@exp-1-23 job_52929778]$ df -Th /scratch
Filesystem     Type  Size  Used Avail Use% Mounted on
/dev/nvme0n1p1 ext4  916G  266M  869G   1% /scratch
[mkandes@exp-1-23 job_52929778]$
```

We see here that the `/scratch` directory is located on an [NVMe](https://en.wikipedia.org/wiki/NVM_Express) drive using the [`ext4`](https://en.wikipedia.org/wiki/Ext4) filesystem. Next try running the `df` command on the `/home` directory, which is where we created our working directory for the tutorial exercices.  

*Command*
```
df -Th /home
```

*Output*
```
[mkandes@exp-1-23 job_52929778]$ df -Th /home
Filesystem     Type    Size  Used Avail Use% Mounted on
/etc/auto.home autofs     0     0     0    - /home
[mkandes@exp-1-23 job_52929778]$ 
```

What is going on here? Let's try and be more specific.

*Command*
```
df -Th "/home/${USER}"
```

*Output*
```
[mkandes@exp-1-23 job_52929778]$ df -Th "/home/${USER}"
Filesystem                        Type  Size  Used Avail Use% Mounted on
10.22.100.112:/pool2/home/mkandes nfs   211T   39T  172T  19% /home/mkandes
[mkandes@exp-1-23 job_52929778]$
```

Your `/home` directory is [automounted](https://en.wikipedia.org/wiki/Automounter)! It's being served from a [Network File System (NFS)](https://en.wikipedia.org/wiki/Network_File_System).

So, to answer the question from above: Q: Why is there such a large difference in the time to download the same dataset when the only thing we've changed is the target directory? A: Downloading the data to the `/scratch` directory used the **local** `/scratch` disk on the compute node, while downloading the data to our working directory in `/home` utilizeed the NFS filesystem, which is a **distributed** network filesystem. 

-  A local [file system](https://en.wikipedia.org/wiki/File_system) is a capability of an operating system that services the applications running on the same computer.[
-  A [distributed file system](https://en.wikipedia.org/wiki/Clustered_file_system#Distributed_file_systems) is a protocol that provides file access between networked computers.

But still, why the difference in performance? 

## Inspect the dataset

Let's take a quick look at the dataset. 

*Command*
```
ls -lahtr CIFAR-10-images/
```

*Output*
```
[mkandes@exp-1-23 job_52929778]$ ls -lahtr CIFAR-10-images/
total 24K
drwx------  3 mkandes root   4.0K Aug  3 15:47 ..
-rw-r--r--  1 mkandes use300   95 Aug  3 15:47 README.md
drwxr-xr-x 12 mkandes use300 4.0K Aug  3 15:47 test
drwxr-xr-x  5 mkandes use300 4.0K Aug  3 15:47 .
drwxr-xr-x 12 mkandes use300 4.0K Aug  3 15:47 train
drwxr-xr-x  8 mkandes use300 4.0K Aug  3 15:47 .git
[mkandes@exp-1-23 job_52929778]$
```

Okay, we see both a `train` and `test` dataset directory. Let's inspect a little further. 

*Command*
```
ls -lahtr CIFAR-10-images/train
```

*Output*
```
[mkandes@exp-1-23 job_52929778]$ ls -lahtr CIFAR-10-images/train
total 1.3M
drwxr-xr-x  5 mkandes use300 4.0K Aug  3 15:47 ..
drwxr-xr-x  2 mkandes use300 128K Aug  3 15:47 airplane
drwxr-xr-x  2 mkandes use300 128K Aug  3 15:47 automobile
drwxr-xr-x  2 mkandes use300 128K Aug  3 15:47 bird
drwxr-xr-x  2 mkandes use300 128K Aug  3 15:47 cat
drwxr-xr-x  2 mkandes use300 128K Aug  3 15:47 deer
drwxr-xr-x  2 mkandes use300 128K Aug  3 15:47 dog
drwxr-xr-x  2 mkandes use300 128K Aug  3 15:47 frog
drwxr-xr-x  2 mkandes use300 128K Aug  3 15:47 horse
drwxr-xr-x  2 mkandes use300 128K Aug  3 15:47 ship
drwxr-xr-x 12 mkandes use300 4.0K Aug  3 15:47 .
drwxr-xr-x  2 mkandes use300 128K Aug  3 15:47 truck
[mkandes@exp-1-23 job_52929778]$
```

Ah. Each dataset directory has category-level directories for the different images. And ... 

*Command*
```
ls -lahtr CIFAR-10-images/train/airplane/
```

*Output*
```
[mkandes@exp-1-23 job_52929778]$ ls -lahtr CIFAR-10-images/train/airplane/
total 20M
...
-rw-r--r--  1 mkandes use300  887 Aug  3 15:47 4992.jpg
-rw-r--r--  1 mkandes use300  818 Aug  3 15:47 4991.jpg
-rw-r--r--  1 mkandes use300  810 Aug  3 15:47 4990.jpg
-rw-r--r--  1 mkandes use300  939 Aug  3 15:47 4989.jpg
-rw-r--r--  1 mkandes use300  813 Aug  3 15:47 4988.jpg
-rw-r--r--  1 mkandes use300  852 Aug  3 15:47 4987.jpg
-rw-r--r--  1 mkandes use300  913 Aug  3 15:47 4986.jpg
-rw-r--r--  1 mkandes use300  924 Aug  3 15:47 4985.jpg
-rw-r--r--  1 mkandes use300  927 Aug  3 15:47 4984.jpg
-rw-r--r--  1 mkandes use300  979 Aug  3 15:47 4983.jpg
-rw-r--r--  1 mkandes use300  912 Aug  3 15:47 4982.jpg
-rw-r--r--  1 mkandes use300  893 Aug  3 15:47 4981.jpg
drwxr-xr-x  2 mkandes use300 128K Aug  3 15:47 .
drwxr-xr-x 12 mkandes use300 4.0K Aug  3 15:47 ..
[mkandes@exp-1-23 job_52929778]$
```

... each category directory holds a lot of raw `*.jpg` images. How many? Use the [`wc`](https://en.wikipedia.org/wiki/Wc_(Unix)) command to get a quick count.

*Command*
```
ls -lahtr CIFAR-10-images/train/airplane/ | wc -l
```

*Output*
```
[mkandes@exp-1-23 job_52929778]$ ls -lahtr CIFAR-10-images/train/airplane/ | wc -l
5003
[mkandes@exp-1-23 job_52929778]$
```

*Command*
```
ls -lahtr CIFAR-10-images/test/airplane/ | wc -l
```

*Output*
```
[mkandes@exp-1-23 job_52929778]$ ls -lahtr CIFAR-10-images/test/airplane/ | wc -l
1003
[mkandes@exp-1-23 job_52929778]$
```

Multiplying by 10 category-level directories, we see that the dataset has approximately 60K raw `*.jpg* image files. Is that a lot? Hint: Think about how much [metadata](https://en.wikipedia.org/wiki/Metadata) may be associated with a large number of files. 

## Zip the Dataset and Copy It Back

Before we close out this interactive session, let's [`zip`](https://www.geeksforgeeks.org/linux-unix/zip-command-in-linux-with-examples) up the dataset directory and copy it back to our working directory.

*Command*
```
zip -r CIFAR-10-images.zip CIFAR-10-images/
```

*Output*
```
[mkandes@exp-1-23 job_52929778]$ zip -r CIFAR-10-images.zip CIFAR-10-images/
  adding: CIFAR-10-images/ (stored 0%)
  adding: CIFAR-10-images/.git/ (stored 0%)
  adding: CIFAR-10-images/.git/packed-refs (deflated 10%)
  adding: CIFAR-10-images/.git/logs/ (stored 0%)
...
  adding: CIFAR-10-images/test/bird/0258.jpg (deflated 20%)
  adding: CIFAR-10-images/test/bird/0203.jpg (deflated 18%)
  adding: CIFAR-10-images/test/bird/0204.jpg (deflated 22%)
[mkandes@exp-1-23 job_52929778]$ 
```

Check that the zip archive file was created successfully.

*Command* 
```
ls -lahtr
```

*Output*
```
[mkandes@exp-1-23 job_52929778]$ ls -lahtr
total 78M
drwxr-xr-x 3 root    root   4.0K Aug  3 15:46 ..
drwxr-xr-x 5 mkandes use300 4.0K Aug  3 15:47 CIFAR-10-images
-rw-r--r-- 1 mkandes use300  78M Aug  3 15:51 CIFAR-10-images.zip
drwx------ 3 mkandes root   4.0K Aug  3 15:51 .
[mkandes@exp-1-23 job_52929778]$
```

And now [`cp`]() it back to your working directory.

*Command*
```
cp CIFAR-10-images.zip ${HOME}/complecs/
```

*Output*
```
[mkandes@exp-9-55 job_48282497]$ cp CIFAR-10-images.zip ${HOME}/complecs/
[mkandes@exp-1-23 job_52929778]$ ls -lahtr ${HOME}/complecs/
total 211M
drwxr-xr-x  2 mkandes use300   10 Jun  4  2009 cifar-10-batches-py
-rw-r--r--  1 mkandes use300 163M Nov 27  2024 cifar-10-python.tar.gz
-rw-r--r--  1 mkandes use300  78M Aug  3 15:53 CIFAR-10-images.zip
drwxr-x--- 24 mkandes use300   43 Aug  3 15:53 ..
-rw-r--r--  1 mkandes use300   89 Aug  3 15:54 cifar-10-python.tar.gz.sha256
-rw-r--r--  1 mkandes use300   57 Aug  3 15:54 cifar-10-python.tar.gz.md5
drwxr-xr-x  3 mkandes use300    7 Aug  3 15:54 .
[mkandes@exp-1-23 job_52929778]$
```

Now that we have a copy of the dataset in our zip archive, we can close our interactive session with the [`exit`](https://en.wikipedia.org/wiki/Exit_(command)) command.

*Command*
```
exit
```

*Output*
```
[mkandes@exp-1-23 job_52929778]$ exit
exit
[mkandes@login01 ~]$
```
# 
[Back to Main Page](https://github.com/sdsc-complecs/data-management)
