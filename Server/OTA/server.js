const dgram = require("dgram")
const client = dgram.createSocket('udp4')
const WebSocket = require('ws');
const fs = require("fs");
const storage = require('node-persist')
const Alert = require("./Alert")
const Board = require("./Board")
const Group = require("./Group")

let pingInterval = null


/**
 * A Board is a representation of an ESP board
 * Every board is keyed on the mac address.
 *
 * @type {{Board}}
 */
let boards = {};
/**
 * Holds all the Groups.
 * @type {{Group}}
 */
let groups = {};

const CLIENTS_PORT = 41222;

/**
 * Sends the go command to the client.
 * @param address - The IP address of the client.
 * @param version - The version of the file that the client should request.
 */
function sendGo(address, version) {
    console.log(CLIENTS_PORT);
    console.log(address);
    client.send(Buffer.from(`go|${version}`), CLIENTS_PORT, address, (err) => {
        if (err)
            console.error(err)
    })
}

/**
 * Sets the on status of a board to true or false
 * @param board
 * @param state {boolean} to indicated the on status of the board
 */
function sendStatus(board, state) {
    client.send(Buffer.from(`on-status|${state}`), CLIENTS_PORT, board.address, (err) => {
        if (err)
            console.error(err)
    })
}

/**
 * Sets the on status of a board to true or false
 * @param board
 * @param state {boolean} to indicated the on status of the board
 */
function sendReset(board) {
    client.send(Buffer.from(`reset`), CLIENTS_PORT, board.address, (err) => {
        if (err)
            console.error(err)
    })
}

function sendIdentify(address) {
    client.send(Buffer.from(`Identify!`), CLIENTS_PORT, address, (err) => {
        if (err)
            console.error(err)
    })
}

function setName(address, name) {
    client.send(Buffer.from(`nm|${name.toString().substr(0, 64)}`), CLIENTS_PORT, address, (err) => {
        if (err)
            console.error(err)
    })
}

function setGroupName(groupID, name) {
    groups[groupID].name = name;
    sendAlert(new Alert(Alert.GROUPNAME, `Group with ID ${groupID} renamed to ${name}`))
    saveGroups();
}

function setGroup(address, groupID) {
    client.send(Buffer.from(`g|${Number(groupID) % 255}`), CLIENTS_PORT, address, (err) => {
        if (err)
            console.error(err)
    })
}

function saveGroups() {
    let tempGroups = {}
    Object.keys(groups).map(key => {
        tempGroups[key] = groups[key].export()
    })
    storage.setItem('groups', tempGroups)
}

/**
 * Removes the alert with the given id from the alert list.
 * @param alertId - the id of the alert to remove.
 */
function clearAlert(alertId) {
    alerts = alerts.filter(a => a.id !== alertId)
}

client.on('error', (err) => {
    console.log(`server error:\n${err.stack}`);
    server.close();
});

/**
 * Listens from the UDP messages send by the clients.
 */
client.on('message', (msg, rinfo) => {
    if (String(msg) !== "Polo")
        console.log(`server got: ${msg} from ${rinfo.address}:${rinfo.port}`);
  
    // register new board
    if (msg.indexOf("|") !== -1) {
        const data = msg.toString().split("|")
        const mac = data[0].split(":").map((x) => Number(x).toString(16)).join(":");
        const name = (data[1] === "") ? "No-Name" : data[1];
        const group = data[2];    
        const version = (data[3] === "") ? "No-Version" : data[3];

        ssid = "No-SSID" 
        dbm = "No signal strength"
     
		if (data.length == 6 ){
           ssid = (data[4] === "") ? "No-SSID" : data[4];
           dbm = (data[5] === "") ? "No signal strength" : data[5];
            }
        
        let updated = false;

        // Check if group exists
        if (typeof groups[group] === "undefined") {
            // Create new group
            groups[group] = new Group(group, `group-${group}`, version)
            saveGroups();
            updated = true;
        }

        // Check if the board exits
        if (typeof boards[mac] === "undefined") {
            // Create new board
            boards[mac] = new Board(mac, group, version,ssid,dbm, name, rinfo.address, client, CLIENTS_PORT)
            // Add to group
            
            groups[group].addBoard(boards[mac])
            // Create board joined alert
            sendAlert(new Alert(Alert.JOINED, `${name} just joined`))
            updated = true;
        } else {
            const board = boards[mac];

            // update time stamp
            boards[mac].lastActiveTime = Date.now();


            // If the board does exist check if it updated
            // Name updated
            if (name !== board.name) {
                sendAlert(new Alert(Alert.UPDATE, `${board.name} is now named ${name}`))
                board.name = name;
                updated = true;
            }

             // Name updated
             if (dbm !== board.dbm) {
               // sendAlert(new Alert(Alert.UPDATE, `${board.name} is now named ${name}`))
                board.dbm = dbm;
               // updated = true;
            }

             // Name updated
             if (ssid !== board.ssid) {
                //sendAlert(new Alert(Alert.UPDATE, `${board.name} is now named ${name}`))
                board.ssid = ssid;
                updated = true;
            }
            
            // Group updated
            if (group !== board.group) {
                sendAlert(new Alert(Alert.UPDATE, `${name} is now in group ${groups[group].name}`))
                // Remove from old group
                groups[board.group].remove(board);
                // Assign to new group
                board.group = group;
                groups[group].addBoard(board);
                updated = true;
            }
            // Version updated
            if (version !== board.version) {
                sendAlert(new Alert(Alert.UPDATE, `${name} is now on version ${version}`))
                board.version = version
                updated = true;
            }
            // Ip updated
            if (rinfo.address !== board.address) {
                sendAlert(new Alert(Alert.UPDATE, `${name} now has address ${rinfo.address}`))
                board.address = rinfo.address
                updated = true;
            }
        }


        // Check if we stole the ip of an existing client.
        for (let [key, value] of Object.entries(boards)) {
            // Skip the current client
            if (key === mac)
                continue;

            // Check if there was an other client with this IP
            if (value.address === rinfo.address) {
                sendAlert(new Alert(Alert.WARNING, `${name} has claimed the ip of ${value.name}. ${value.name} must have disconnected!`))
                // remove old client
                delete boards[key];
                updated = true;
            }
        }

        if (updated)
            updateClients();
    }

    if (String(msg) === "Polo") {
        const matches = Object.values(boards).filter((board) => {
            return board.address === rinfo.address;
        })
        if (matches.length === 0) {
            // Board is not known ask to identify
            sendIdentify(rinfo.address)
        } else {
            // Update time of life.
            matches[0].lastActiveTime = Date.now();
        }
    }
});

client.on('listening', async () => {
    // Start the storage modual
    await storage.init({
        dir: './resources/storage/',
        stringify: JSON.stringify,
        parse: JSON.parse,
        encoding: 'utf8',
        logging: false,  // can also be custom logging function
        ttl: false,
        expiredInterval: 2 * 60 * 1000, // every 2 minutes the process will clean-up the expired cache
        forgiveParseErrors: false
    });

    // Load known Groups
    const g = await storage.getItem('groups')
    if (g) {
        for (const key of Object.keys(g)) {
            const group = g[key]
            groups[key] = new Group(group.id, group.name, group.version)
        }
    }

    // Announce that the server is awake
    const address = client.address();
    console.log('server listening online!');
    console.log('Web portal is available on: http://localhost:3000')
    console.log(`Board connection on port ${address.port}`)
    console.log('Websockets on port 8080')

    // Prevent infinite intervals
    if (pingInterval !== null){
        clearInterval(pingInterval)
        pingInterval = null
    }

    // Send the broadcast message.
    pingInterval = setInterval(function () {
        client.setBroadcast(true);
        client.send(Buffer.from(`Marco`), 41222, "255.255.255.255");
        setTimeout(checkAllBoards,1000)
    }, 5000)
});

function checkAllBoards() {
    let change = false;
    for (const board of Object.values(boards)) {
        let alive = board.checkAlive()
        if (alive === 0) {
            change =true;
            sendAlert(new Alert(Alert.OFFLINE, `${board.name} is now offline!`))
        } else if (alive === 1) {
            change=true
            sendAlert(new Alert(Alert.ONLINE, `${board.name} is now back online!`))
        }
    }
    if (change) {
        updateClients()
    }
}

client.bind(41234);

// -------------------------------------------------------------------
// Websocket stuff
const wss = new WebSocket.Server({port: 8080});
const clients = {};

let counter = 0;

wss.on('connection', function connection(ws) {
    ws.ID = counter
    counter++

    ws.on('message', function incoming(message) {
        console.log('received: %s', message);
    });

    clients[ws.ID] = ws

    ws.on('close', function () {
        delete clients[ws.ID]
    })

    setTimeout(() => {
        ws.send("B|" + JSON.stringify({
            Boards: boards,
            Groups: groups
        }))
        fs.readdir(`${__dirname}/../files/`, (err, files) => {
            if (err) {
                console.error(err)
                return
            }
            files = files.map((file) => {
                return {
                    file: file,
                    time : fs.statSync(`${__dirname}/../files/${file}`).mtimeMs
                }
            })
            ws.send("F|" + JSON.stringify(files))
        })
    }, 1000)

});

const alertBacklog = []

function sendAlert(alert) {
    if (Object.values(clients).length === 0)
        return alertBacklog.push(alert)

    let alerts = []

    if (alert)
        alerts.push(alert)

    while (alertBacklog.length > 0) {
        alerts.push(alertBacklog.pop())
    }

    for (const client of Object.values(clients)) {
        for (const a of alerts) {
            client.send("A|" + JSON.stringify(a))
        }
    }
}

function updateClients() {
    for (const client of Object.values(clients)) {
        client.send("B|" + JSON.stringify({
            Boards: boards,
            Groups: groups
        }))
    }
}

function updateFileList() {
    fs.readdir(`${__dirname}/../files/`, (err, files) => {
        if (err) {
            console.error(err)
            return
        }
        // add data to file
        files = files.map((file) => {
            return {
                file: file,
                time : fs.statSync(`${__dirname}/../files/${file}`).ctime
            }
        })

        for (const client of Object.values(clients)) {
            client.send("F|" + JSON.stringify(files))
        }
    })
}

function deleteGroup(groupId) {
    delete groups[groupId];
}

module.exports = {sendReset, sendStatus, sendGo, boards, groups, clearAlert, setName, setGroup, setGroupName, updateFileList, updateClients, saveGroups, deleteGroup};
