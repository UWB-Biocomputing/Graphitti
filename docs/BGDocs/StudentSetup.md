# Student Quickstart
Prerequisites:
- [Husky OnNet (BIG-IP Edge Client)](https://www.lib.washington.edu/help/connect/husky-onnet) to connect to the UW network from off campus
- [Access to the Graphitti servers](#using-graphitti-servers)
- [Visual Studio Code](https://code.visualstudio.com/)

Use Husky OnNet to connect to the server "UW Campus Network Traffic Only."

Other external guides for remote VS Code development:
- [Connecting VS Code to UW Servers](http://depts.washington.edu/cssuwb/wiki/write_high_quality_c_code#visual_studio_code)
- [Visual Studio Code: Remote (C++) Development on Linux Lab Machines](http://depts.washington.edu/cssuwb/wiki/vsc_and_remote_development)
- [Remote Development using SSH](https://code.visualstudio.com/docs/remote/ssh). 

## Connecting VS Code to UW Server
1. Install the [Remote Development extension pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) either from the web or through the VS Code extensions marketplace. This is a pack of three related extensions.

    If you run into problems, see [https://code.visualstudio.com/docs/remote/troubleshooting](https://code.visualstudio.com/docs/remote/troubleshooting). In particular, you may need to set `"remote.SSH.useLocalServer": false` in VSC's settings.json file if you run into connection problems.
   
2. Click the Remote Explorer icon on the left sidebar. Add new in remote explorer: raiju.uwb.edu or otachi.uwb.edu (or both).
   
3. It will ask if you want to update your config file. Go to the config file and add a line with `User` and your uw net id to the file. The config file should now look like this: 

    ``` 
    Host raiju.uwb.edu
      HostName raiju.uwb.edu
      User myuwnetidhere
    ```

    At this point you should be connected to Husky OnNet. 
    
    [See Windows Troubleshooting if you have trouble connecting here.](#windows-troubleshooting)

4. Your configured SSH target(s) should be listed on the Remote Explorer icon. Right click a target and connect to host.

## Setting up Graphitti
1. In your NETID folder, clone the Graphitti repository:

    ```
    $ git clone https://github.com/UWB-Biocomputing/Graphitti
    ```

2. Click Open folder and click Graphitti. You should now have the directory on VS Code. To see what version you're on, use: 

    ```
    $ git status
    ```

    To run:

    ```
    $ cd build

    $ cmake .. 
    ```

    This will generate a makefile. Then type:

    ```
    $ make
    ```

    You can then run a selection of sample tests to ensure the simulator is able to run:

    ```
    $ ./tests
    ```

    If ` $ ./tests` fails, start with this command instead of the cmake command: 

    ```
    $ cmake -D CMAKE_CXX_COMPILER=/opt/rh/devtoolset-8/root/usr/bin/g++ .

    $ make

    $ ./tests

    $ ./cgraphitti -c ../configfiles/test-small.xml
    ```

## Using Visual Studio Code
Install the [C/C++ extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) for IntelliSense and debugging. We recommend setting the IntelliSense Cache Size setting to 10 MB. The default size is 5120 MB, which can result in VSC consuming all of your allotted space on the CSS Linux Machines.

### Building
VSC can be configured to compile from CMake so that you don't have to type build and launch commands into the terminal every time you want to run. 
- See [https://code.visualstudio.com/docs/cpp/cmake-linux](https://code.visualstudio.com/docs/cpp/cmake-linux) and follow along using the existing CMakeLists.txt in the project's root.
- In your launch.json file, you'll want to configure the program, args, and cwd options as such:
    ```json
    "program": "${workspaceFolder}/build/cgraphitti",
    "args": ["-c ../configfiles/test-tiny.xml"],
    "cwd": "${workspaceFolder}/build",
    ```
    with the args changing depending on the arguments you actually want to pass.

### Other Visual Studio Code Resources
- Prof. Pisan has a video in which he sets up VSC: https://youtu.be/CrLpiV-KMbM
- Here's an intro video on using VSC: https://youtu.be/fnPhJHN0jTE

## Create an SSH key pair
To eliminate the need to type your password in to log in to the CSS Linux machines, consider using public/private key authentication. 
- The ssh-keygen program on Mac/Linux can be used to generate a public/private key pair. For instructions on key generation, see this [tutorial on GitHub](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). This is also a good time to add an SSH key to your GitHub account if you haven't already.
- If you put the public key in ~/.ssh/authorized_keys on the Linux machine, and both keys in ~/.ssh on your local machine, then this type of authentication should work. 
- Make sure to guard your private key!

## Using Graphitti Servers
UW students can use the school's research servers to run Graphitti. The raiju and otachi servers have computational NVIDIA GPUs optimized for floating point value computations. More info on the hardware can be found [here](http://depts.washington.edu/cssuwb/wiki/computing_resources#raijuuwbedu). If the user preference is to run Graphitti locally, the CUDA libraries are not necessary; the CPU version of the simulator is user-friendly on local machines.

To access these machines:

1. Contact Dr. Michael Stiber at [stiber@uw.edu](mailto:stiber@uw.edu) and express reason for gaining access to servers.
   
2. Download Husky OnNet: https://www.lib.washington.edu/help/connect/husky-onnet

3. Login with UW NetID
   
4. Connect through comand line SSH or use Visual Studio Code.

## Windows Troubleshooting
If you have trouble connecting to the server through VS Code, your ssh exe file might not be properly fetched from VS Code. If you have trouble with this step, refer to these links: 

- https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse

- https://fred151.net/site/2018/09/23/how-to-install-openssh-on-windows-7-10/#:~:text

You can download the files here: https://github.com/PowerShell/Win32-OpenSSH/releases

Put it under C:\Windows\System32\OpenSSH
