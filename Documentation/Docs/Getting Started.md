
# Getting Started

[BrainGrid](https://github.com/UWB-Biocomputing/BrainGrid) and

 - List item

[WorkBench](https://github.com/UWB-Biocomputing/WorkBench)

Workbench is used to understand whether previously-generated results are still valid and how they might or might not be comparable to new results.

###

1.  Visual Studio Code can be used to open and edit files on a remote Linux machine. This includes opening a terminal inside VSC to compile and run. Go to [https://code.visualstudio.com/](https://code.visualstudio.com/)  to download VSC, and to [https://code.visualstudio.com/docs/remote/ssh](https://code.visualstudio.com/docs/remote/ssh)  to get the instructions for the "Visual Studio Code Remote Development Extension Pack". If you run into problems, see [https://code.visualstudio.com/docs/remote/troubleshooting](https://code.visualstudio.com/docs/remote/troubleshooting). In particular, you may need to set "remote.SSH.useLocalServer": false  in VSC's settings.json  file if you run into connection problems.

2.  To eliminate the need to type your password in to log in to the CSS Linux machines, consider using public/private key authentication. The  ssh-keygen  program on Mac/Linux can be used to generate a public/private key pair. If you put the public key in  ~/.ssh/authorized_keys  on the Linux machine, and both keys in  ~/.ssh  on your local machine, then this type of authentication should work. Make sure to guard your private key!

3.  Prof.  Stiber  is working with the CSS system staff to get git updated to version 2 on raiju and otachi.

> Written with [StackEdit](https://stackedit.io/).
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNzgyODMzNiwtMjY5OTY5ODM0XX0=
-->