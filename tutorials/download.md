# CIFAR through the tubes: Downloading data from the internet

<img src='../images/usdci2025.jpg' width='100%' height='100%'/>

[Image Credit: B. Roberts, NLR](https://research-hub.nlr.gov/en/publications/data-center-infrastructure-in-the-united-states-november-2025-2/)

The aim of this tutorial is to introduce you to command-line tools that are useful for downloading data from the internet and verifing the data is correct. The dataset we'll be working with is the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), a well-known machine learning dataset that consists of 60K 32x32 colour images broken out into 10 classes, with 6000 images per class.

Let's get started by logging into Expanse with your account either directly via `ssh` and the [Expanse User Portal](https://portal.expanse.sdsc.edu)

*Command*
```
ssh mkandes@login.expanse.sdsc.edu
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
