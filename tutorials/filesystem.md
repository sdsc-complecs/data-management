# More files, more problems: Advantages and limitations of different filesystems

<img src='https://wiki.lustre.org/images/a/a3/Lustre_File_System_Overview_%28DNE%29_lowres_v1.png' width='100%' height='100%'/>

[Image Credit: Malcom, wiki.lustre.org](https://wiki.lustre.org)

The aim of this tutorial is to teach you about the advantages and limitations of different [filesystems](https://en.wikipedia.org/wiki/File_system) that you'll typically find available to you on a high-performance computing system.

## Login

If you are not already logged in to Expanse, please login to Expanse with your account either directly via `ssh` or through the [Expanse User Portal](https://portal.expanse.sdsc.edu)

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

## Clone the Dataset

Once logged, go ahead and try to clone this GitHub repository that contains a copy of the CIFAR-10 dataset. However, please be prepared to **cancel** the download. 
