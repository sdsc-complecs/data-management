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

Before we answer the question above, let's try to clone the dataset to an alternative location on Expanse. To make your life simpler, define the following [`alias`](https://en.wikipedia.org/wiki/Alias_(command)) command to start an interactive session on one of Expanse's compute nodes. 

*Command*
```
alias start-interactive='srun --partition=shared --account=sdp157 --reservation=complecs --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=4G --time=00:30:00 --pty --wait=0 /bin/bash'
```

*Output*
```
[mkandes@login02 complecs]$ alias start-interactive='srun --partition=shared --account=sdp157 --reservation=complecs --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --mem=4G --time=00:30:00 --pty --wait=0 /bin/bash'
[mkandes@login02 complecs]$
```

Then start an interactive session.

*Command*

```
start-interactive
```

*Output*
```
[mkandes@login02 complecs]$ start-interactive
srun: job 48282497 queued and waiting for resources
srun: job 48282497 has been allocated resources
[mkandes@exp-9-55 complecs]$
```

Once your interactive session starts, navigate to your `/scratch` working directory. 

*Command*
```
cd "/scratch/${USER}/job_${SLURM_JOB_ID}"
```

*Output*
```
[mkandes@exp-9-55 complecs]$ cd "/scratch/${USER}/job_${SLURM_JOB_ID}"
[mkandes@exp-9-55 job_48282497]$ pwd
/scratch/mkandes/job_48282497
[mkandes@exp-9-55 job_48282497]$ ls -lahtr
total 8.0K
drwxr-xr-x 3 root    root 4.0K Apr 21 06:49 ..
drwx------ 2 mkandes root 4.0K Apr 21 06:49 .
[mkandes@exp-9-55 job_48282497]$
```

Now, try and clone the dataset here. What do you observe?

*Command*
```
time -p git clone https://github.com/YoongiKim/CIFAR-10-images.git
```

*Output*
```
[mkandes@exp-9-55 job_48282497]$ time -p git clone https://github.com/YoongiKim/CIFAR-10-images.git
Cloning into 'CIFAR-10-images'...
remote: Enumerating objects: 60027, done.
remote: Total 60027 (delta 0), reused 0 (delta 0), pack-reused 60027 (from 1)
Receiving objects: 100% (60027/60027), 19.94 MiB | 42.45 MiB/s, done.
Resolving deltas: 100% (59990/59990), done.
real 2.26
user 0.67
sys 0.96
[mkandes@exp-9-55 job_48282497]$
```

Why is there such a large difference in the time to download the same dataset when the only thing we've changed is the target directory? Hint: What type of underlying filesystems are in use? You can check basic filesystem information with the [`df`](https://en.wikipedia.org/wiki/Df_(Unix)) command. Start by running the `df` command on the `/scratch` directory.

*Command*
```
df -Th /scratch
```

*Output*
```
[mkandes@exp-9-55 job_48282497]$ df -Th /scratch/
Filesystem     Type  Size  Used Avail Use% Mounted on
/dev/nvme0n1p1 ext4  916G  268M  869G   1% /scratch
[mkandes@exp-9-55 job_48282497]$
```

We see here that the `/scratch` directory is located on an [NVMe](https://en.wikipedia.org/wiki/NVM_Express) drive using the [`ext4`](https://en.wikipedia.org/wiki/Ext4) filesystem. Next try running the `df` command on the `/home` directory, which is where we created our working directory for the tutorial exercices.  

*Command*
```
df -Th /home
```

*Output*
```
[mkandes@exp-9-55 job_48282497]$ df -Th /home
Filesystem     Type    Size  Used Avail Use% Mounted on
/etc/auto.home autofs     0     0     0    - /home
[mkandes@exp-9-55 job_48282497]$
```

What is going on here? Let's try and be more specific.

*Command*
```
df -Th "/home/${USER}"
```

*Output*
```
[mkandes@exp-9-55 job_48282497]$ df -Th "/home/${USER}"
Filesystem                        Type  Size  Used Avail Use% Mounted on
10.22.100.112:/pool2/home/mkandes nfs   212T   37T  176T  18% /home/mkandes
[mkandes@exp-9-55 job_48282497]$
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
[mkandes@exp-9-55 job_48282497]$ ls -lahtr CIFAR-10-images/
total 24K
-rw-r--r--  1 mkandes use300   95 Apr 21 06:50 README.md
drwxr-xr-x 12 mkandes use300 4.0K Apr 21 06:50 test
drwxr-xr-x  5 mkandes use300 4.0K Apr 21 06:50 .
drwxr-xr-x 12 mkandes use300 4.0K Apr 21 06:50 train
drwxr-xr-x  8 mkandes use300 4.0K Apr 21 06:50 .git
drwx------  3 mkandes root   4.0K Apr 21 06:59 ..
[mkandes@exp-9-55 job_48282497]$
```

Okay, we see both a `train` and `test` dataset directory. Let's inspect a little further. 

*Command*
```
ls -lahtr CIFAR-10-images/train
```

*Output*
```
[mkandes@exp-9-55 job_48267623]$ ls -lahtr CIFAR-10-images/train/
total 1.2M
drwxr-xr-x  5 mkandes use300 4.0K Apr 20 19:23 ..
drwxr-xr-x  2 mkandes use300 120K Apr 20 19:23 airplane
drwxr-xr-x  2 mkandes use300 120K Apr 20 19:23 automobile
drwxr-xr-x  2 mkandes use300 120K Apr 20 19:23 bird
drwxr-xr-x  2 mkandes use300 120K Apr 20 19:23 cat
drwxr-xr-x  2 mkandes use300 120K Apr 20 19:23 deer
drwxr-xr-x  2 mkandes use300 120K Apr 20 19:23 dog
drwxr-xr-x  2 mkandes use300 120K Apr 20 19:23 frog
drwxr-xr-x  2 mkandes use300 120K Apr 20 19:23 horse
drwxr-xr-x  2 mkandes use300 120K Apr 20 19:23 ship
drwxr-xr-x 12 mkandes use300 4.0K Apr 20 19:23 .
drwxr-xr-x  2 mkandes use300 120K Apr 20 19:23 truck
[mkandes@exp-9-55 job_48267623]$
```

Ah. Each dataset directory has category-level directories for the different images. And ... 

*Command*
```
ls -lahtr CIFAR-10-images/train/airplane/
```

*Output*
```
[mkandes@exp-9-55 job_48282497]$ ls -lahtr CIFAR-10-images/train/airplane/
total 20M
-rw-r--r--  1 mkandes use300  873 Apr 21 06:50 0067.jpg
-rw-r--r--  1 mkandes use300  784 Apr 21 06:50 0066.jpg
-rw-r--r--  1 mkandes use300  887 Apr 21 06:50 0065.jpg
...
-rw-r--r--  1 mkandes use300  828 Apr 21 06:50 4947.jpg
-rw-r--r--  1 mkandes use300  877 Apr 21 06:50 4946.jpg
-rw-r--r--  1 mkandes use300  869 Apr 21 06:50 4945.jpg
drwxr-xr-x  2 mkandes use300 120K Apr 21 06:50 .
drwxr-xr-x 12 mkandes use300 4.0K Apr 21 06:50 ..
[mkandes@exp-9-55 job_48282497]$
```

... each category directory holds a lot of raw `*.jpg` images. How many? Use the [`wc`](https://en.wikipedia.org/wiki/Wc_(Unix)) command to get a quick count.

*Command*
```
ls -lahtr CIFAR-10-images/train/airplane/ | wc -l
```

*Output*
```
[mkandes@exp-9-55 job_48282497]$ ls -lahtr CIFAR-10-images/train/airplane/ | wc -l
5003
[mkandes@exp-9-55 job_48282497]$
```

*Command*
```
ls -lahtr CIFAR-10-images/test/airplane/ | wc -l
```

*Output*
```
[mkandes@exp-9-55 job_48282497]$ ls -lahtr CIFAR-10-images/test/airplane/ | wc -l
1003
[mkandes@exp-9-55 job_48282497]$
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
[mkandes@exp-9-55 job_48282497]$ zip -r CIFAR-10-images.zip CIFAR-10-images/
  adding: CIFAR-10-images/ (stored 0%)
  adding: CIFAR-10-images/.git/ (stored 0%)
  adding: CIFAR-10-images/.git/packed-refs (deflated 10%)
  adding: CIFAR-10-images/.git/logs/ (stored 0%)
...
  adding: CIFAR-10-images/test/bird/0258.jpg (deflated 20%)
  adding: CIFAR-10-images/test/bird/0203.jpg (deflated 18%)
  adding: CIFAR-10-images/test/bird/0204.jpg (deflated 22%)
[mkandes@exp-9-55 job_48282497]$
```

Check that the zip archive file was created successfully.

*Command* 
```
ls -lahtr
```

*Output*
```
[mkandes@exp-9-55 job_48282497]$ ls -lahtr
total 78M
drwxr-xr-x 3 root    root   4.0K Apr 21 07:21 ..
drwxr-xr-x 5 mkandes use300 4.0K Apr 21 07:21 CIFAR-10-images
-rw-r--r-- 1 mkandes use300  78M Apr 21 07:31 CIFAR-10-images.zip
drwx------ 3 mkandes root   4.0K Apr 21 07:31 .
[mkandes@exp-9-55 job_48282497]$
```

And now [`cp`]() it back to your working directory.

*Command*
```
cp CIFAR-10-images.zip ${HOME}/complecs/
```

*Output*
```
[mkandes@exp-9-55 job_48282497]$ cp CIFAR-10-images.zip ${HOME}/complecs/
[mkandes@exp-9-55 job_48282497]$ ls -lahtr ${HOME}/complecs/
total 211M
drwxr-xr-x  2 mkandes use300   10 Jun  4  2009 cifar-10-batches-py
-rw-r--r--  1 mkandes use300 163M Jun  4  2009 cifar-10-python.tar.gz
-rw-r--r--  1 mkandes use300   57 Apr 20 15:59 cifar-10-python.tar.gz.md5
-rw-r--r--  1 mkandes use300   89 Apr 20 16:04 cifar-10-python.tar.gz.sha256
drwxr-xr-x  5 mkandes use300    6 Apr 20 16:37 CIFAR-10-images
drwxr-x--- 29 mkandes use300   46 Apr 20 16:53 ..
drwxr-xr-x  4 mkandes use300    8 Apr 21  2026 .
-rw-r--r--  1 mkandes use300  78M Apr 21  2026 CIFAR-10-images.zip
[mkandes@exp-9-55 job_48282497]$
```

Now that we have a copy of the dataset in our zip archive, we can close our interactive session with the [`exit`](https://en.wikipedia.org/wiki/Exit_(command)) command.

*Command*
```
exit
```

*Output*
```
[mkandes@exp-9-55 job_48282497]$ exit
exit
[mkandes@login02 complecs]$
```
# 
[Back to Main Page](https://github.com/sdsc-complecs/data-management)
