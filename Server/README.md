# OTA SERVER

# prerequisites
This server runs on Node.js 1.12 and above. To run and install it, you will need both node and npm.
To install these, check out the website of node.js. https://nodejs.org/en/. From this website you can install the `LTS` version of node. \
The node Installation should include `npm`.
For additional support check out https://docs.npmjs.com/downloading-and-installing-node-js-and-npm#using-a-node-installer-to-install-nodejs-and-npm.


# Simple start
If you have all the prerequisites then you should be able to just run the `start` file. For **Windows** this is `start.bat` and for **Mac and Linux** this is `start.sh`

If this doesn't start the server then please check out the rest of this documentation.

# Setup

## step 1
To install the servers dependencies, open the servers location (the folder teh server files reside in) in a terminal and run the following command
```shell
npm install
```
You can ignore possible notices and warnings presented during the installation.

On Windows it could be that this fails, if so it could be that you are missing the building tools essentials. \
if this is the case, follow this tutorial to fix that https://www.npmjs.com/package/windows-build-tools

### How to open folder in terminal.
#### Mac
Open a new terminal. In this terminal type the following: `cd ` (don't forget the space after `cd`) then drag the folder from finder into the terminal.

#### Windows
In explorer hold shift while pressing right mouse button in a folder. Then from the menu select `Open `

## step 2
Once everything is installed create a folder called `files` inside the server folder.

# Running the server
To run the server, open a terminal on the location of the server and run the following command.
```shell
npm run start
```

The server should now indicate that it started listening for OTA devices.

To open the server page, go http://127.0.0.1:3000/ in a browser. You should now see the server.

If you open the web page and it says


# updating server
You can update the server by overwriting all server files with the new ones. There is no need to remove file on your own.
Once all files that needed to be overwritten are overwritten run the following command again:
```shell
npm install
```
