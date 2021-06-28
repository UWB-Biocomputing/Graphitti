TODO: Make sure this is password or uwnet id protected. 
# Student Quickstart
Ensure you have husky onnet
https://www.lib.washington.edu/help/connect/husky-onnet

Connect to the server "UW Campus Network Traffic Only"  

Connecting VS Code to UW Servers: in depth
http://depts.washington.edu/cssuwb/wiki/write_high_quality_c_code

http://depts.washington.edu/cssuwb/wiki/vsc_and_remote_development

Ensure you have visual studio code 

## Connecting VS Code to UW Server
1. Go to extensions tab in VS Code. Search "Remote SSH" and select this: Remote - SSH ms-vscode-remote.remote-ssh
   
2. add new in remote explorer: raiju.uwb.edu or otachi.uwb.edu
   
3. It will ask if you want to update your config file Go to the config file and add a line with `User` and your uw net id to the file. The config file should now look like this: 

``` 
Host raiju.uwb.edu
  HostName raiju.uwb.edu
  User myuwnetidhere
  ```

4. At this point you should be connected to big ip. 

/// Just for windows troubleshooting ///
if you have trouble connecting to the server through vs code, your ssh exe file might not be properly fetched from vs code. If you have trouble with this step, refer to these links: https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse
https://fred151.net/site/2018/09/23/how-to-install-openssh-on-windows-7-10/#:~:text

You can download the files here: https://github.com/PowerShell/Win32-OpenSSH/releases

put it under C:\Windows\System32\OpenSSH

continuing ... 
```
$ git clone https://github.com/UWB-Biocomputing/Graphitti
```

click Open folder and click Graphitti

you should now have the directory on vs code

to see what version you're on, use: 
```
$ git status
```

To run:

```
$ cd build

$ cmake .. 
```

this will generate a makefile. Then type:

```
$ make
```

You can then run a selection of sample tests to ensure the simulator is able to run:

```
$ ./tests
```

if ` $ ./tests` fails, start with this command instead of the cmake command: 

```
$ cmake -D CMAKE_CXX_COMPILER=/opt/rh/devtoolset-8/root/usr/bin/g++ .

$ make

$ ./tests

$ ./graphitti -c ../configfiles/test-small.xml
```








## Using Graphitti Servers

1. Must contact Dr. Michael Stiber at [stiber@uw.edu](mailto:stiber@uw.edu) and express reason for gaining access to servers.
   
2. Download Husky OnNet: https://www.lib.washington.edu/help/connect/husky-onnet

3. Login with UW NetID
   
4. The UW servers have a computational NVIDIA GPU optimized for floating point value computations. If the user preference is to run Graphitti locally, the CUDA libraries are not necessary; the CPU version of the simulator is user-friendly on local machines.

## Using Visual Studio Code
1.  Visual Studio Code can be used to open and edit files on a remote Linux machine. This includes opening a terminal inside VSC to compile and run. Go to [https://code.visualstudio.com/](https://code.visualstudio.com/)  to download VSC, and to [https://code.visualstudio.com/docs/remote/ssh](https://code.visualstudio.com/docs/remote/ssh)  to get the instructions for the "Visual Studio Code Remote Development Extension Pack". If you run into problems, see [https://code.visualstudio.com/docs/remote/troubleshooting](https://code.visualstudio.com/docs/remote/troubleshooting). In particular, you may need to set "remote.SSH.useLocalServer": false  in VSC's settings.json  file if you run into connection problems.

2.  To eliminate the need to type your password in to log in to the CSS Linux machines, consider using public/private key authentication. The  ssh-keygen  program on Mac/Linux can be used to generate a public/private key pair. If you put the public key in  ~/.ssh/authorized_keys  on the Linux machine, and both keys in  ~/.ssh  on your local machine, then this type of authentication should work. Make sure to guard your private key!



## Connect Server to Git
1. use git clone 


